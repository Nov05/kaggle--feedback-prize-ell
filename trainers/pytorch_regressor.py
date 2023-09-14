import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

## local imports
from torch_utils import SkleanDataset
from config import MSFTDeBertaV3Config
from trainers.base_trainers import SklearnTrainer



## final activation layer for the ELL task
## ELL - Kaggle competition: Feedback Prize - English Language Learning
class ELLActivation(nn.Module):
	def __init__(self, force_half_points=False):
		super().__init__()
		self._force_half_points = force_half_points

	def forward(self, x):
		y = torch.sigmoid(x) * 4. + 1.
		if self._force_half_points:
			y = torch.round(y * 2) / 2
		return y



##############################################################
## fully connected layer(s) + activation layer(s)
##############################################################
class SequentialNeuralNetwork(nn.Module):
	def __init__(self, X, y, hidden_dims=None, n_hidden=3, force_half_points=False):
		## if hidden_dims is not None, it will override n_hidden
		assert hidden_dims is not None or n_hidden is not None, \
			"Hint: Either n_hidden or hidden_dims should not be None."

		super(SequentialNeuralNetwork, self).__init__()

		## parameters
		self._force_half_points = force_half_points
		self._model = nn.Sequential()
		self._input_dim = X.shape[1]
		self._output_dim = y.shape[1]
		
		print("model info:")
		if hidden_dims: ## hidden layer dimensions; if hidden_dims isn't None, it will override n_hidden
			print("hidden_dims:", hidden_dims)
			self._hidden_dims = hidden_dims
			self._n_hidden = len(self._hidden_dims)
		elif n_hidden: ## hidden layer number
			print("n_hidden:", n_hidden)
			self._n_hidden = n_hidden
			self._alpha = (np.log(self._output_dim) / np.log(self._input_dim)) ** (1 / (self._n_hidden + 1))
			self._hidden_dims = [
				int(np.round(self._input_dim ** (self._alpha ** i))) for i in np.arange(1, self._n_hidden + 1)]

		if self._n_hidden > 0:
			for dim_in, dim_out in zip([self._input_dim] + self._hidden_dims, self._hidden_dims):
				# linear_layer = nn.init.xavier_uniform(linear_layer.weight)
				linear_layer = nn.Linear(dim_in, dim_out, bias=True)
				self._model.append(linear_layer)
				self._model.append(nn.ReLU())
			self._model.append(nn.Linear(self._hidden_dims[-1], self._output_dim, bias=True))
		else:
			self._model.append(nn.Linear(self._input_dim, self._output_dim, bias=True))
		self._model.append(ELLActivation(force_half_points=self._force_half_points))
		print(self._model)

	def forward(self, x):
		return self._model(x)



##############################################################
## using a simple vanilla neural network model
##############################################################
class NNTrainer(SklearnTrainer):
	def __init__(self,
				 fastext_model_path=None,
				 deberta_config:MSFTDeBertaV3Config=None,
				 target_columns=None,
				 feature_columns=None,
				 train_file_name=None,
				 test_file_name=None,
				 submission_file_name = None):
		super().__init__(fastext_model_path=fastext_model_path,
				         deberta_config=deberta_config,
			             target_columns=target_columns,
						 feature_columns=feature_columns,
						 train_file_name=train_file_name,
						 test_file_name=test_file_name,
						 submission_file_name=submission_file_name)
		## pytorch specific
		self._optimizer = None
		self._loss_fn = nn.MSELoss()
		self._loss_values = dict()
		self._training_device = self._deberta_config.training_device
		self._inference_device = self._deberta_config.inference_device
		print(f"this torch regressor will be trained on {self._training_device}")


	def get_data_loader(self, X, y, bactch_size, shuffle=True):
		data = SkleanDataset(X, y)
		data_loader = DataLoader(dataset=data,
							 	 batch_size=bactch_size,
								 shuffle=shuffle,
								 collate_fn=lambda x: tuple(x_.to(self._training_device) for x_ in default_collate(x))
		)
		return data_loader


	def train(self, X, y, params):
		## instantiate the model
		self._model = SequentialNeuralNetwork(X, y,
											  hidden_dims=params["hidden_dims"],
											  n_hidden=params["n_hidden"],
											  force_half_points=params["force_half_points"])
		self._model.to(self._training_device)
		self._optimizer = torch.optim.Adam(self._model.parameters(),
			                               lr=params["learning_rate"])
		self._loss_values["train"]=[]
		if params["with_validation"]:
			print("using validation")
			self._loss_values["val"] = []
			X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["val_size"])
			train_data_loader = self.get_data_loader(X_train, y_train, params["batch_size"], params["shuffle"])
			val_data_loader = self.get_data_loader(X_val, y_val, params["batch_size"], params["shuffle"])
		else:
			print("no validation")
			train_data_loader = self.get_data_loader(X, y, params["batch_size"], params["shuffle"])

		## start training
		for _ in range(params["num_epochs"]):
			epoch_loss_values = dict(train=[])

			for X_train, y_train in train_data_loader:
				## compute prediction error
				y_pred_train = self._model(X_train.to(self._training_device))
				train_loss = self._loss_fn(y_pred_train, y_train.to(self._training_device))
				epoch_loss_values["train"].append(train_loss.item())

				## backpropagation
				self._optimizer.zero_grad()
				train_loss.backward()
				self._optimizer.step()
			self._loss_values["train"].append(np.sqrt(np.mean(epoch_loss_values["train"])))

			if params["with_validation"]:
				epoch_loss_values["val"] = []
				self._model.eval()  ## optional when not using Model Specific layer
				for X_val, y_val in val_data_loader:
					## forward pass
					y_pred_val = self._model(X_val)
					## find loss
					val_loss = self._loss_fn(y_pred_val, y_val)
					## calculate loss
					epoch_loss_values["val"].append(val_loss.item())
				self._loss_values["val"].append(np.sqrt(np.mean(epoch_loss_values["val"])))

		print("training completed")


	def plot_loss_values(self):
		print("plotting losses...")
		fig, ax = plt.subplots(figsize=(20, 5))
		ax.plot(np.array(self._loss_values["train"]), label="training")
		if "val" in self._loss_values.keys():
			ax.plot(np.array(self._loss_values["val"]), label="validation")
		ax.set_xlabel("Epochs")
		ax.set_ylabel("Losses")
		ax.legend()
		plt.show()


	def predict(self, X, recast_scores=True):
		self._model.to(self._inference_device)
		y_pred = self._model(SkleanDataset.csr_to_torch(X).to(self._inference_device)).cpu().detach().numpy()
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred