import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import torch
from torch.utils.data import DataLoader
from torch_utils import MeanPooling
import fasttext
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

## local imports
from config import MSFTDeBertaV3Config
from english_utils import clean_special_characters


def ft_langdetect(text: str, fasttext_model):
	labels, scores = fasttext_model.predict(text)
	label = labels[0].replace("__label__", '')
	score = min(float(scores[0]), 1.0)
	return {"lang": label,
		    "score": score}


def ftlangdetect_english_score(series: pd.Series, fasttext_model) -> np.array:
	""""""
	res = series.apply(lambda x: ft_langdetect(clean_special_characters(x), fasttext_model))
	return res.apply(lambda x: x['score'] if x["lang"] == 'en' else 1 - x['score']).values.reshape(-1, 1)


class FTLangdetectTransformer(BaseEstimator, TransformerMixin):
	def __init__(self,
			     model_path):
		"""
			This transformer outputs the predictions
			from a pretrained fasttext langdetect model
		"""
		self._model_path = model_path
		self._model = fasttext.load_model(self._model_path)


	def __repr__(self):
		return f"FTLangdetectTransformer from path '{self._model_path}'"


	def ft_langdetect(self, text: str):
		labels, scores = self._model.predict(text)
		label = labels[0].replace("__label__", '')
		score = min(float(scores[0]), 1.0)
		return {"lang": label,
			    "score": score}


	def fit(self, X, y=None):
		return self


	def transform(self, series: pd.Series) -> np.array:
		check_is_fitted(self, ['_model'])
		res = series.apply(lambda x: self.ft_langdetect(clean_special_characters(x)))
		return res.apply(lambda x: x['score'] if x["lang"] == 'en' else 1 - x['score']).values.reshape(-1, 1)


class PooledDeBertaTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, config):
		self.config:MSFTDeBertaV3Config = config
		self.tokenizer = AutoTokenizer.from_pretrained(self.config._model_path)
		self.model = AutoModel.from_pretrained(self.config._model_path).to(self.config.inference_device)


	def fit(self, series, y=None):
		return self


	def prepare_input(self, input_texts):
		"""
			this will truncate the longest essays.
			Consider adding some sliding window logic and pooling/aggregation to capture the full text?
		"""
		if isinstance(input_texts, pd.Series):
			texts_list = input_texts.values.tolist()
		else:
			texts_list = input_texts
		inputs = self.tokenizer.batch_encode_plus(
			texts_list,
			return_tensors=None,
			add_special_tokens=True,
			max_length=self.config.tokenizer_max_length,
			pad_to_max_length=True,
			truncation=True
		)
		for k,v in inputs.items():
			inputs[k] = torch.tensor(v, dtype=torch.long)
		return inputs


	@torch.no_grad()
	def feature(self, inputs):
		self.model.eval()
		# last_hidden_states = self.model(**{k: v.to(self.config._inference_device) for k, v, in inputs.items()})[0]
		last_hidden_states = self.model(
			**{k: v.to(self.config.training_device) for k, v, in inputs.items()}
		).last_hidden_state
		feature = MeanPooling()(
			last_hidden_states,
			inputs['attention_mask'].to(self.config.training_device)
		)
		return feature


	def batch_transform(self, series):
		y_preds_list = []
		data_loader = DataLoader(dataset=series,
								 batch_size=self.config._batch_size,
								 shuffle=False)
		for _series in tqdm(data_loader):
			_inputs = self.prepare_input(_series).to(self.config.inference_device)
			with torch.no_grad():
				__y_preds = self.feature(_inputs)
			y_preds_list.append(__y_preds.cpu())
		y_preds = np.concatenate(y_preds_list)
		return y_preds


	def simple_transform(self, series):
		inputs = self.prepare_input(series).to(self.config.inference_device)
		y_preds = self.feature(inputs).to(self.config.inference_device)
		return y_preds


	def transform(self, series):
		# check_is_fitted(self, ['model', 'tokenizer'])
		if self.config._batch_transform:
			print("using batch transform")
			return self.batch_transform(series)
		else:
			print("using simple transform")
			return self.simple_transform(series)


	def fit_transform(self, series, y=None):
		self.fit(series, y=None)
		y_preds = self.transform(series)
		return y_preds
