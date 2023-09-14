import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod

## local imports
from config import TRAIN_FILE_PATH, TEST_FILE_PATH, SUBMISSION_FILE_PATH, \
				   FEATURE_COLUMNS, TARGET_COLUMNS, \
				   MSFTDeBertaV3Config
from pipelines import make_features_pipeline



"""
inheritance relationship:
ModelTrainer 
    |--> SklearnTrainer
            |--> NNTrainer
			|--> SklearnRegressorTrainer
	|--> DebertaTrainer
"""



class ModelTrainer(ABC):
	def __init__(self,
			     feature_columns=None,
				 target_columns=None,
				 train_file_name=None,
				 test_file_name=None,
				 submission_file_name=None):
		self._feature_columns = feature_columns if feature_columns else FEATURE_COLUMNS
		self._target_columns = target_columns if target_columns else TARGET_COLUMNS
		self._train_file_name = train_file_name if train_file_name else TRAIN_FILE_PATH
		self._test_file_name = test_file_name if test_file_name else TEST_FILE_PATH
		self._submission_file_name = submission_file_name if submission_file_name else SUBMISSION_FILE_PATH
		self._model = None


	def __repr__(self):
		return "'ModelTrainer' object"
 

	def load_data(self, file_name=None, is_test=False):
		if not file_name:
			if not is_test: ## train data
				df = pd.read_csv(self._train_file_name)
			else: ## test data
				df = pd.read_csv(self._test_file_name)
		else:
			df = pd.read_csv(file_name)
		if not is_test:
			for column in self._target_columns:
				assert df[column].dtype==float
		return df


	@staticmethod
	def recast_scores(y_pred):
		y_pred = np.round(y_pred * 2) / 2
		y_pred = np.min([y_pred, 5.0*np.ones(y_pred.shape)], axis=0)
		y_pred = np.max([y_pred, 1.0*np.ones(y_pred.shape)], axis=0)
		return y_pred


	@abstractmethod
	def train(self, X, y):
		pass


	def predict(self, X, recast_scores=True):
		y_pred = self._model.predict(X)
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred


	@staticmethod
	def evaluate(y_true, y_pred):
		assert y_true.shape==y_pred.shape
		return np.mean([mean_squared_error(y_true[:, idx], y_pred[:, idx], squared=False) \
			            for idx in range(y_true.shape[1])])


	def evaluate_per_category(self, y_true, y_pred):
		eval_dict = dict()
		for idx, score_name in enumerate(self._target_columns):
			eval_dict[score_name] = mean_squared_error(y_true[:, idx], y_pred[:, idx], squared=False)
		return eval_dict


	@abstractmethod
	def test(self, recast_scores=True, write_file=True):
		pass


	def make_submission_file(self, submission_df, y_pred_submission):
		submission_df[self._target_columns] = y_pred_submission
		submission_df = submission_df[['text_id'] + self._target_columns]
		print(f"writing submission to: '{self._submission_file_name}'")
		submission_df.to_csv(self._submission_file_name, index=False)
	


## for model types: 'nn'(nueral network), 'lgb', 'xgb', 'linear", 'dummy'(mean)
class SklearnTrainer(ModelTrainer):
	def __init__(self,
			  	 fastext_model_path=None,
				 deberta_config:MSFTDeBertaV3Config=None,
				 feature_columns=None,
				 target_columns=None,
				 train_file_name=None,
				 test_file_name=None,
				 submission_file_name=None):
		super().__init__(feature_columns=feature_columns,
						 target_columns=target_columns,
						 train_file_name=train_file_name,
						 test_file_name=test_file_name,
						 submission_file_name=submission_file_name)
		## feature extraction pipeline
		self._fastext_model_path = fastext_model_path
		self._deberta_config = deberta_config
		self._pipeline = make_features_pipeline(fastext_model_path=self._fastext_model_path,
										        deberta_config=self._deberta_config)
		

	def get_training_set(self, df):
		df_features = df[self._feature_columns]
		y = df[self._target_columns].values
		return df_features, y


	@staticmethod
	def split_data(df_features, y, test_size, random_state=42):
		df_features_train, df_features_test, y_train, y_test \
			= train_test_split(df_features, y, test_size=test_size, random_state=random_state)
		return df_features_train, df_features_test, y_train, y_test
	

	def test(self, recast_scores=True, write_file=True):
		print(f"loading test data from: '{self._test_file_name}'")
		submission_df = super().load_data(is_test=True)
		print(f"transforming testing data...")
		X_submission = self._pipeline.transform(submission_df)
		y_pred_submission = self.predict(X_submission, recast_scores=recast_scores)
		if recast_scores:
			y_pred_submission = self.recast_scores(y_pred_submission)
		if write_file:
			super().make_submission_file(submission_df, y_pred_submission)
		return y_pred_submission