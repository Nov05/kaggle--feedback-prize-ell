import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset 

## local imports
from config import FEATURE_COLUMNS, TARGET_COLUMNS



class SkleanDataset(Dataset):
	@classmethod
	def csr_to_torch(cls, X):
		if isinstance(X, csr_matrix):
			_X = X.todense()
		else:
			_X = X
		Xt = torch.from_numpy(_X.astype(np.float32))
		return Xt
	

	def __init__(self, X, y):
		self.X = self.csr_to_torch(X)
		self.y = torch.from_numpy(y.astype(np.float32))
		self.len = self.X.shape[0]


	def __getitem__(self, index):
		return self.X[index], self.y[index]


	def __len__(self):
		return self.len



## convert a dataframe to tokenized inputs and targets
class EssayDataset(Dataset):
    def __init__(self, df, config, tokenizer=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.feature_columns = FEATURE_COLUMNS
        self.target_columns = TARGET_COLUMNS
        self.max_len = config.max_length
        self.tokenizer = tokenizer
        self.is_test = is_test


    def __getitem__(self, idx):
        sample = self.df.loc[idx, self.feature_columns[0]]
        tokenized = self.tokenizer.encode_plus(sample,
                                               None,
                                               add_special_tokens=True,
                                               max_length=self.max_len,
                                               truncation=True,
                                               padding='max_length')
        inputs = {
            "input_ids": torch.tensor(tokenized['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        }
        if self.is_test:
            return inputs
        else:
            labels = self.df.loc[idx, self.classes].to_list()
            targets = {"labels": torch.tensor(labels, dtype=torch.float32)}
            return inputs, targets


    def __len__(self):
        return len(self.df)
	


class MeanPooling(nn.Module):  
	def __init__(self, clamp_min=1e-9):
		super().__init__()
		self._clamp_min = clamp_min


	def forward(self, last_hidden_state, attention_mask):
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
		sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
		sum_mask = input_mask_expanded.sum(1)
		sum_mask = torch.clamp(sum_mask, min=self._clamp_min)
		mean_embeddings = sum_embeddings / sum_mask
		return mean_embeddings
	


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings



class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings



## There may be a bug in my implementation because it does not work well.
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average