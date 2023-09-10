from trainers.sklearn_regressor import SklearnRegressorTrainer
from trainers.pytorch_regressor import NNTrainer

import warnings
warnings.filterwarnings("ignore")

## local imports
from config import MSFTDeBertaV3Config, \
	               DEBERTAV3BASE_MODEL, FASTTEXT_MODEL_PATH, \
				   TRAINING_PARAMS, \
				   MODEL_TYPE, TEST_SIZE


#################################################################################
##
##    This repo is dedicated to the following NLP task, aka. ELL or FB3
##    Kaggle competition: Feedback Prize - English Language Learning
##
#################################################################################
## https://www.kaggle.com/competitions/feedback-prize-english-language-learning


if __name__ == "__main__":


	## microsfot deberta v3 base model configuration
	deberta_config = MSFTDeBertaV3Config(
		model_name=DEBERTAV3BASE_MODEL,
		# pooling="mean",         ## "mean" is the only option for now
		# training_device="cuda", ## use default device, cuda or mps
		# inference_device="cpu", ## use default device. note: for the efficiency competition part, gpu use is not allowed
		batch_transform=True,
		batch_size=10
	)


	## there are 2 types of models available as regressor: 
	## 1. lightgbm/xgboost/linear/dummy
	## 2. simple vanilla nueral network
	model_type = MODEL_TYPE ## options: 'lgb', 'xgb', 'linear", 'dummy' (col mean), 'nn'
	print(f"model type: {model_type}")
	if model_type=='nn':
		model_trainer = NNTrainer(fastext_model_path=FASTTEXT_MODEL_PATH,
								  deberta_config=deberta_config)
	else:
		model_trainer = SklearnRegressorTrainer(model_type=model_type,
												fastext_model_path=FASTTEXT_MODEL_PATH,
												deberta_config=deberta_config)
	

	print("loading training data...")
	df = model_trainer.load_data() 
	df_features, y = model_trainer.get_training_set(df.iloc[:50,:]) ## y is a 6-col numpy.ndarray
	df_features_train, df_features_test, y_train, y_test = \
		model_trainer.split_data(df_features, y, test_size=TEST_SIZE) ## types: df, df, np array, np array


	## check the function make_features_pipeline() in pipelines.py for details
	## basically there are 3 types of features: manually extracted, fasttext embeddings, deberta meanpooling output
	print("transforming training data...")
	X_train = model_trainer._pipeline.fit_transform(df_features_train)


    ## feed the features into xgboost/lightgbm/etc. or a simple vanilla neural network
	print("training...")
	model_trainer.train(X_train, y_train, TRAINING_PARAMS[model_type])


	print("testing...")
	X_test = model_trainer._pipeline.transform(df_features_test)
	y_pred = model_trainer.predict(X_test)


	print("evaluating...")
	print(model_trainer.evaluate(y_test, y_pred))

	if model_type=='nn':
		print("ploting losses...")
		model_trainer.plot_loss_values()

	print("making submission file...")
	model_trainer.make_submission_df()

	print("all done")