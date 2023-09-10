import os
import torch
import platform


################################################
## training hyperparameters
################################################
MODEL_TYPE = 'nn' ## options: 'lgb', 'xgb', 'linear", 'dummy' (col mean), 'nn'
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

## e.g. hidden_dims=[6]
##      num_epochs=750 can reach MCRMSE=0.48440331870350567
##      1000, 0.4866859875915251 (probably overfitted a bit)
## [16], 100, 0.47854053637866856, a wider single hidden layer seems to work well
TRAINING_PARAMS["nn"] = dict(
	hidden_dims=[64], ## if hidden_dims is not None, it will override n_hiddden
	n_hidden=None, ## hidden layer numbers, in/out will be calculated automatically
	num_epochs=200,
	batch_size=512,
	shuffle=True,
	learning_rate=0.0001,
	with_validation=True,
	val_size=0.25,
	training_device=None, ## None will lead to GPU if available. options: None, "cpu"
	force_half_points=False,
)


################################################
## paths
################################################
PLATFORM_ARM64 = 'arm64' ## mac etc.
PLATFORM_WIN = 'Win' ## Windows
ROOT_DIR = "." if PLATFORM_ARM64 in platform.platform() or PLATFORM_WIN in platform.platform() \
			   else "/kaggle"
INPUT_DIR = "input" ## Kaggle's \kaggle\input dir is read-only
CHALLENGE_NAME = "feedback-prize-english-language-learning" ## Kaggle's train, test, and sample submission data folder
SUBMISSION_DIR = "working" ## Kaggle's output folder
MODELS_DIR = os.path.join(ROOT_DIR, INPUT_DIR, "models")
## lid.176.ftz is the compressed version of the model, with a file size of 917kB.
## https://fasttext.cc/docs/en/language-identification.html
FASTTEXT_MODEL_PATH = os.path.join(MODELS_DIR, "fasttext", "lid.176.ftz")


def get_default_device():
	if PLATFORM_ARM64 in platform.platform():
		return "mps:0"
	elif torch.cuda.is_available():
		return "cuda:0"
	return "cpu"


################################################
## Microsoft DeBerta V3 model configuration
################################################
class MSFTDeBertaV3Config:
	def __init__(self,
			     model_name,
			     pooling="mean",
			     training_device=None,
				 inference_device=None,
			     batch_transform=True,
			     batch_size=100):
		"""
		manage the deberta model configuration
		"""
		assert pooling=="mean", "We removed all other implementations other than 'mean'."

		self._model_name = model_name
		self.pooling = pooling
		self.training_device = training_device if training_device \
			                   else get_default_device()
		self.inference_device = inference_device if inference_device \
		                        else get_default_device()
		self._batch_transform = batch_transform
		self._batch_size = batch_size
		self._models_dir = MODELS_DIR
		self._model_path = os.path.join(self._models_dir, self._model_name)
		self.gradient_checkpointing = False
		self.tokenizer_max_length = 512
		print(self)

	@property
	def config(self):
		return os.path.join(self._model_path, "config")

	@property
	def model(self):
		return os.path.join(self._model_path, "model")

	@property
	def tokenizer(self):
		return os.path.join(self._model_path, "tokenizer")


	def __repr__(self):
		return f"""MSFTDeBertaV3Config object:
\t model name: {self._model_name}
\t traning device: {self.training_device}
\t inference device: {self.inference_device}"""