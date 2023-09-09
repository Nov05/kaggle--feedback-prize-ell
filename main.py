from trainers.sklearn_regressor import SklearnRegressorTrainer
from trainers.pytorch_regressor import NNTrainer

import warnings
warnings.filterwarnings("ignore")

## local imports
from config import MSFTDeBertaV3Config, \
	               FASTTEXT_MODEL_PATH


TEST_SIZE = 0.33

TRAINING_PARAMS = dict()
TRAINING_PARAMS["dummy"] = dict()
TRAINING_PARAMS["linear"] = dict()
TRAINING_PARAMS["xgb"] = dict(
	booster='gbtree',
	colsample_bylevel=1.0,
	colsample_bytree=1.0,
	gamma=0.0,
	learning_rate=0.1,
	max_delta_step=0.0,
	max_depth=7,
	in_child_weights=1.0,
	n_estimators=50, #100
	normalize_type='tree',
	num_parallel_tree=1,
	n_jobs=1,
	objective='reg:squarederror',
	one_drop=False,
	rate_drop=0.0,
	reg_alpha=0.0,
	reg_lambda=1.0,
	sample_type='uniform',
	silent=True,
	skip_drop=0.0,
	subsample=1.0
)

TRAINING_PARAMS["lgb"] = dict(
	boosting_type='gbdt',
	num_leaves=15, #31
	max_depth=14, #-1
	learning_rate=0.05,
	n_estimators=50, #100
	subsample_for_bin=200000,
	objective=None,
	min_split_gain=0.1,
	min_child_weight=0.001,
	min_child_samples=10,
	subsample=0.9, #1.0
	subsample_freq=0,
	colsample_bytree=1.0,
	reg_alpha=0.0,
	reg_lambda=0.0,
	random_state=None,
	n_jobs=None,
	importance_type='split',
#     verbosity=1
)

BATCH_SIZE = 512

TRAINING_PARAMS["nn"] = dict(
	hidden_dims=[6],
	n_hidden=None,
	batch_size=BATCH_SIZE,
	force_half_points=False,
	num_epochs=2000,
	learning_rate=0.0001,
	shuffle=True,
	val_size=0.25,
	with_validation=True,
	training_device=None #"cpu"
)


if __name__ == "__main__":

	deberta_config = MSFTDeBertaV3Config(
		model_name="deberta-v3-base",
		# pooling="mean", ## "mean" is the only option for now
		# training_device="mps", ## use default device
		# inference_device="cpu",		
		batch_transform=True,
		batch_transform_size=10
	)

	model_type = 'nn' ## 'lgb', 'xgb', 'linear", 'dummy', 'nn'
	if model_type=='nn':
		model_trainer = NNTrainer(fastext_model_path=FASTTEXT_MODEL_PATH,
								deberta_config=deberta_config)
	else:
		model_trainer = SklearnRegressorTrainer(model_type=model_type,
												fastext_model_path=FASTTEXT_MODEL_PATH,
												deberta_config=deberta_config)
	
	print("loading training data...")
	df = model_trainer.load_data() 
	df_features, y = model_trainer.get_training_set(df) #(df.iloc[:100,:]) ## y is a 6-col numpy.ndarray
	df_features_train, df_features_test, y_train, y_test = \
		model_trainer.split_data(df_features, y, test_size=TEST_SIZE) ## types: df, df, np array, np array

	print("transforming data...")
	X_train = model_trainer._pipeline.fit_transform(df_features_train)

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

	print("all done")