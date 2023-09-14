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
        self.classes = TARGET_COLUMNS
        self.max_len = config['max_length']
        self.tokenizer = tokenizer
        self.is_test = is_test


    def __getitem__(self,idx):
        sample = self.df['full_text'][idx]
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