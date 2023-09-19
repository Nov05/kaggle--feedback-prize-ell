import os
import sys
import warnings
warnings.filterwarnings("ignore")

## local imports
from config import TRAINING_PARAMS, TEST_SIZE, \
                   MSFTDeBertaV3Config, \
	               DEBERTAV3BASE_MODEL_PATH, FASTTEXT_MODEL_PATH, \
				   DEBERTA_FINETUNED_MODEL_PATH, TARGET_COLUMNS
from trainers.sklearn_regressor import SklearnRegressorTrainer
from trainers.pytorch_regressor import NNTrainer
from trainers.deberta_trainer import DebertaTrainer

os.environ['TOKENIZERS_PARALLELISM']='true'



################################################################################
"""
    This repo is dedicated to the following NLP task, aka. ELL or FB3
	Kaggle competition: Feedback Prize - English Language Learning
"""
################################################################################
## https://www.kaggle.com/competitions/feedback-prize-english-language-learning



if __name__ == "__main__":

	## $python <model_type> <recast_scores>
	##     e.g. $python main.py deberta 0
	try:
		if sys.argv[1]: model_type = sys.argv[1]
	except:
		model_type = 'deberta'
	print(f"model type: {model_type}")
	try:
		if sys.argv[2]: recast_scores = bool(int(sys.argv[2]))
	except:
		recast_scores = False
	print(f"recast scores: {recast_scores}")


	## there are 2 types of skelearn models available as regressor: 
	##     1. lightgbm/xgboost/linear/dummy: 'lgb', 'xgb', 'linear", 'dummy'(mean)
	##     2. simple vanilla nueral network: 'nn'(nueral network)
	## there is 1 type of deberta model available
	##     1. deberta-vs-base finetuned model: 'deberta'
	if model_type in ['nn', 'lgb', 'xbg', 'linear', 'dummy']:
		## microsoft deberta v3 base model configuration
		deberta_config = MSFTDeBertaV3Config(
			model_name='deberta-v3-base',
			model_path=DEBERTAV3BASE_MODEL_PATH,
			# pooling="mean",          ## "mean" is the only option for this model
			# training_device="cuda",  ## use default device, cuda or mps or cpu (you won't want to use cpu for training lol)
			# inference_device="cuda", ## use default device. note: for the efficiency competition part, gpu use is not allowed
			batch_transform=True,
			batch_size=8
		)

	## create model trainer
	if model_type=='deberta':
		model_trainer = DebertaTrainer(model_path=DEBERTA_FINETUNED_MODEL_PATH, 
								       is_test=False)
	elif model_type=='nn':
		model_trainer = NNTrainer(fastext_model_path=FASTTEXT_MODEL_PATH,
								  deberta_config=deberta_config)
	else: ## 'lgb', 'xgb', 'linear", 'dummy'(mean)
		model_trainer = SklearnRegressorTrainer(model_type=model_type,
												fastext_model_path=FASTTEXT_MODEL_PATH,
												deberta_config=deberta_config)
		
	## train the model	
	if model_type in ['nn', 'lgb', 'xbg', 'linear', 'dummy']:
		print("loading training data...")
		df = model_trainer.load_data() 
		df_features, y = model_trainer.get_training_set(df) #(df.iloc[:10,:]) ## y is a 6-col numpy.ndarray
		df_features_train, df_features_test, y_train, y_test = \
			model_trainer.split_data(df_features, y, test_size=TEST_SIZE) ## types: df, df, np array, np array

		## check the function make_features_pipeline() in pipelines.py for details
		## basically there are 3 types of features: manually extracted, fasttext embeddings, deberta meanpooling output
		print("transforming training data...")
		X_train = model_trainer._pipeline.fit_transform(df_features_train)

		## feed the features into xgboost/lightgbm/etc. or a simple vanilla neural network
		print("training...")
		model_trainer.train(X_train, y_train, TRAINING_PARAMS[model_type])

		print("evaluating...")
		X_test = model_trainer._pipeline.transform(df_features_test)
		y_pred = model_trainer.predict(X_test)
		print(model_trainer.evaluate(y_test, y_pred))
		if model_type=='nn':
			model_trainer.plot_loss_values()

	## inference
	model_trainer.test(recast_scores=recast_scores)

	# ## test on train data to debug high mcrmse issue, 
	# ## which was caused by the data loader shuffling of the data
	# print(f"testing on train data...")
	# y_pred = model_trainer.test(data_loader=model_trainer.train_loader,
	# 						    recast_scores=False, 
	# 							write_submission_file=False)
	# y_true = model_trainer.train_dataset.df[TARGET_COLUMNS].to_numpy()
	# print(f"y_true: {type(y_true)}, {y_true.shape}")
	# print(f"y_pred: {type(y_pred)}, {y_pred.shape}")
	# print(f"train data mcrmse: {model_trainer.evaluate(y_true, y_pred)}")
	
	print("all done")