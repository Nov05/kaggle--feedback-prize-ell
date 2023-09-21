import os
import torch
import platform



## constants
FEATURE_COLUMNS = ["full_text"]
TARGET_COLUMNS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


################################################
## paths
################################################
## local or kaggle, etc.
PLATFORM_ARM64 = "arm64" ## mac etc.
PLATFORM_WIN = "Win" ## Windows

## kaggle dir, local folder structure simulates it
CHALLENGE_NAME = "feedback-prize-english-language-learning" ## Kaggle's train, test, and sample submission data folder
ROOT_DIR = "." if PLATFORM_ARM64 in platform.platform() or PLATFORM_WIN in platform.platform() \
			   else "/kaggle"
INPUT_DIR = "input" ## Kaggle's \kaggle\input dir is read-only
WORKING_DIR = "working" ## Kaggle's working dir 20GB, outputs are stored here
SUBMISSION_DIR = WORKING_DIR 

## data files
TRAIN_FILE_PATH = os.path.join(ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "train.csv")
TEST_FILE_PATH = os.path.join(ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "test.csv")
SUBMISSION_FILE_PATH = os.path.join(ROOT_DIR, SUBMISSION_DIR, "submission.csv")

## kaggle input datasets dir, this repo and models as datasets will be attached to it
DATASETS_DIR = os.path.join(ROOT_DIR, INPUT_DIR)
FASTTEXT_MODEL_PATH = os.path.join(DATASETS_DIR, "fasttextmodel", "lid.176.ftz")
DEBERTAV3BASE_MODEL_PATH = os.path.join(DATASETS_DIR, "microsoftdeberta-v3-base", "deberta-v3-base")
DEBERTA_FINETUNED_MODEL_PATH1 = os.path.join(DATASETS_DIR, "models", "00_40epochs.pth")
DEBERTA_FINETUNED_MODEL_PATH2 = os.path.join(DATASETS_DIR, "models", "_fold0_best.pth")
DEBERTA_FINETUNED_CONFIG_PATH = os.path.join(DATASETS_DIR, "models", "config.pth")

"""
Kaggle directories
	data files:
		/kaggle/input/<challenge name>/sample_submission.csv
		/kaggle/input/<challenge name>/train.csv
		/kaggle/input/<challenge name>/test.csv
		/kaggle/input/<attach different datasets here>
	output folder:
		/kaggle/working/
"""
## Input data files are available in the read-only "../input/" directory
## You can write up to 20GB to the current directory (/kaggle/working/) 
## that gets preserved as output when you create a version using "Save & Run All" 
## You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

## lid.176.bin, which is faster and slightly more accurate, but has a file size of 126MB ;
## lid.176.ftz is the compressed version of the model, with a file size of 917kB.
## https://fasttext.cc/docs/en/language-identification.html



def get_default_device():
	if PLATFORM_ARM64 in platform.platform():
		return "mps:0"
	elif torch.cuda.is_available():
		return "cuda:0"
	return "cpu"



################################################
## deberta-v3-base embedding as a feature extractor
################################################
class MSFTDeBertaV3Config:
	def __init__(self,
			  	 using_deberta,
			     model_name,
				 model_path,
			     pooling="mean",
			     training_device=None,
				 inference_device=None,
			     batch_transform=True,
			     batch_size=10):
		"""
			manage the deberta model configuration
		"""
		assert pooling=="mean", "We removed all other implementations other than 'mean'."

		self.using_deberta = using_deberta
		self._model_name = model_name
		self._model_path = model_path
		self.pooling = pooling
		self.training_device = training_device if training_device \
			                   else get_default_device()
		self.inference_device = inference_device if inference_device \
		                        else get_default_device()
		self._batch_transform = batch_transform
		self._batch_size = batch_size
		self.gradient_checkpointing = False
		self.tokenizer_max_length = 512
		print(self)


	def __repr__(self):
		return f"""MSFTDeBertaV3Config object:
\t model name: {self._model_name}
\t traning device: {self.training_device}
\t inference device: {self.inference_device}"""
	


################################################
## training hyperparameters
################################################
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

## e.g. hidden_dims=[6], num_epochs=750 can reach MCRMSE=0.48440331870350567
##      1000, 0.4866859875915251 (probably overfitted a bit)
##      [16], 100, 0.47854053637866856, a wider single hidden layer seems to work well
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

## for fine-turning the deberta-v3-base model EssayModel
TRAINING_PARAMS["deberta"] = {
	'model': DEBERTAV3BASE_MODEL_PATH, ## 'microsoft/deberta-v3-base'
    'dropout': 0.5,
    'max_len': 512,
    'batch_size': 8, ## any number larger than 8 causes in CUDA OOM for unfreezed encoder on Kaggle GPU
    'epochs': 7,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6, # 1e-8 default
    'freeze_encoder': True,
	'test_size': TEST_SIZE,
}



class CustomeDebertaModelConfig:
	## model construction
	model_name = "microsoft/deberta-v3-base"
	model_path = DEBERTAV3BASE_MODEL_PATH
	finetuned_model_path = DEBERTA_FINETUNED_MODEL_PATH2
	finetuned_config_path = DEBERTA_FINETUNED_CONFIG_PATH
	pooling = 'attention' ## options: mean, max, min, attention, weightedlayer
	target_columns = TARGET_COLUMNS
	num_targets = len(target_columns)

	## training
	freeze = True
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_size, num_workers = 8, 4
	num_epochs, print_freq = 5, 20 ## print every 20 steps when training    
	save_all_models = False ## save model at end of every epoch
    
	loss_func = 'SmoothL1' ## 'SmoothL1', 'RMSE'
	gradient_checkpointing = True
	gradient_accumulation_steps = 1
	max_grad_norm = 1000 ## gradient clipping
	apex = True ## whether using Automatic Mixed Precision 
    
    ## layerwise learning rate decay
	layerwise_lr, layerwise_lr_decay = 5e-5, 0.9
	layerwise_weight_decay = 0.01
	layerwise_adam_epsilon = 1e-6
	layerwise_use_bertadam = False
    
	scheduler = 'cosine'
	num_cycles, num_warmup_steps= 0.5, 0
	encoder_lr, decoder_lr, min_lr  = 2e-5, 2e-5, 1e-6
	max_len = 512
	weight_decay = 0.01
    
	fgm = True ## whether using FGM (Fast Gradient Method) adversarial training algorith
	adv_lr, adv_eps, eps, betas = 1, 0.2, 1e-6, (0.9, 0.999)
	unscale = True
    
	## Multilabel Stratified K-Fold
	n_fold = 4
	trn_fold = list(range(n_fold))

	seed = 42 ## random seed
	debug = False ## debug mode，using only a few training data, n_fold=2，epoch=2
	wandb = False ## whether using Weights & Biaes to log training information
	
	# OUTPUT_DIR = f"./{model_name.replace('/', '-')}/"
	# train_file = '../input/feedback-prize-english-language-learning/train.csv'
	# test_file = '../input/feedback-prize-english-language-learning/test.csv'
	# submission_file = '../input/feedback-prize-english-language-learning/sample_submission.csv'	
	output_dir = WORKING_DIR
	train_file = TRAIN_FILE_PATH
	test_file = TEST_FILE_PATH
	submission_file = SUBMISSION_FILE_PATH


class CFG:
	model_name = "microsoft/deberta-v3-base"
	model_path = DEBERTAV3BASE_MODEL_PATH

	batch_size, n_targets, num_workers = 8, 6, 4 ## batch_size=8 kaggle upper limit
	target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
	epochs, print_freq = 5, 20 # 训练时每隔20step打印一次    
	save_all_models = False # 是否每个epoch都保存数据
	gradient_checkpointing = True

	loss_func = 'SmoothL1' # 'SmoothL1', 'RMSE'
	pooling = 'attention' # mean, max, min, attention, weightedlayer
	gradient_checkpointing = True
	gradient_accumulation_steps = 1 # 是否使用梯度累积更新
	max_grad_norm = 1000 #梯度裁剪
	apex = True # 是否进行自动混合精度训练 

	# 启用llrd
	layerwise_lr,layerwise_lr_decay = 5e-5,0.9
	layerwise_weight_decay = 0.01
	layerwise_adam_epsilon = 1e-6
	layerwise_use_bertadam = False

	scheduler = 'cosine'
	num_cycles ,num_warmup_steps= 0.5,0
	encoder_lr,decoder_lr,min_lr  = 2e-5,2e-5 ,1e-6
	max_len = 512
	weight_decay = 0.01

	fgm = True # 是否使用fgm对抗网络攻击
	wandb=False
	adv_lr,adv_eps,eps,betas = 1,0.2,1e-6,(0.9, 0.999)
	unscale =True

	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

	seed=42
	n_fold=4
	trn_fold=list(range(n_fold))
	debug=False # debug表示只使用少量样本跑代码，且n_fold=2，epoch=2

	OUTPUT_DIR = f"./{model_name.replace('/', '-')}/"
	train_file = '../input/feedback-prize-english-language-learning/train.csv'
	test_file = '../input/feedback-prize-english-language-learning/test.csv'
	submission_file = '../input/feedback-prize-english-language-learning/sample_submission.csv'