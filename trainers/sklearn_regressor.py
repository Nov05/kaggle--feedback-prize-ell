import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

## local imports
from config import MSFTDeBertaV3Config
from trainers.base_trainers import SklearnTrainer



##############################################################
## one of such models: dummy (mean), linear, XGBoost, LightGBM
##############################################################
class SklearnRegressorTrainer(SklearnTrainer):

	def __init__(self,
			  	 model_type='lgb', ## different regressor: 'lgb', 'xgb', 'linear', 'dummy'(mean)
                 fastext_model_path=None,
				 deberta_config:MSFTDeBertaV3Config=None,
				 target_columns=None,
				 feature_columns=None,
				 train_file_name=None,
				 test_file_name=None,
				 submission_file_name=None):
		super().__init__(fastext_model_path=fastext_model_path,
				         deberta_config=deberta_config,
			             target_columns=target_columns,
						 feature_columns=feature_columns,
						 train_file_name=train_file_name,
						 test_file_name=test_file_name,
						 submission_file_name=submission_file_name)
		self._model_type = model_type

	def train(self, X, y, params=None):
		if self._model_type == "lgb":
			print("creating LightGBM regressor...")
			self._model = MultiOutputRegressor(lgb.LGBMRegressor(**params if params else {}))
		elif self._model_type == "xgb":
			print("creating XGBoost regressor...")
			self._model = MultiOutputRegressor(xgb.XGBRegressor(**params if params else {}))
		elif self._model_type == "linear":
			print("creating linear model...")
			self._model = LinearRegression()
		elif self._model_type == "dummy":
			print("creating dummy model...")
			self._model = DummyRegressor(strategy="mean")
		else:
			raise ValueError("unknown model type!")
		self._model.fit(X, y)


	def predict(self, X, recast_scores=True):
		y_pred = self._model.predict(X)
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred