import os
import torch
import platform


PLATFORM_ARM64 = 'arm64' ## mac etc.
PLATFORM_WIN = 'Win' ## Windows
ROOT_DIR = "." if PLATFORM_ARM64 in platform.platform() or PLATFORM_WIN in platform.platform() \
			   else "/kaggle"
INPUT_DIR = "input" ## \kaggle\input is read-only
CHALLENGE_NAME = "feedback-prize-english-language-learning" ## kaggle train, test, and sample submission data
SUBMISSION_DIR = "working" ## kaggle output folder
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


class MSFTDeBertaV3Config:
	def __init__(self,
			     model_name,
			     pooling="mean",
			     training_device=None,
				 inference_device=None,
			     batch_transform=True,
			     batch_transform_size=100):
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
		self._batch_transform_size = batch_transform_size
		self._models_dir = MODELS_DIR
		self._model_path = os.path.join(self._models_dir, self._model_name)
		self.gradient_checkpointing = False
		self.tokenizer_max_length = 512

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