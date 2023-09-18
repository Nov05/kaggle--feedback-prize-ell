import torch
from torch import nn
from transformers import AutoModel, AutoConfig

## local imports
from config import FB3Config
from torch_utils import MeanPooling, MaxPooling, MinPooling, \
                        AttentionPooling, WeightedLayerPooling



class EssayModel(nn.Module):
    """
        EssayModel = deberta-v3-base (frozen) + mean-pooling + 2 fully-connected layers
    """
    def __init__(self, 
                 config:dict, 
                 n_targets=6):
        super().__init__()

        self.model_name = config['model']
        self.freeze = config['freeze_encoder']

        ## freeze the original model
        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        ## attach mean-pooling and fully-connected layers
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(config['dropout']) ## dropout 
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, 64) ## 1 fully connected layer is enought
        self.fc2 = nn.Linear(64, n_targets)


    def forward(self, inputs):
        outputs = self.encoder(**inputs, return_dict=True) ## kwarg expansion is not supported by torch script
        outputs = self.pooler(outputs['last_hidden_state'], inputs['attention_mask'])
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs
    


class FB3Model(nn.Module):
    def __init__(self, 
                 config, # a class
                 is_finetuned=True):  
        super().__init__()

        self.fb3config = fb3config
        if is_finetuned:
            ## load model from .pth
            self.model = AutoModel.from_pretrained(self.fb3config.finetuned_model_path) 
            ## load model config
            if FB3Config.model_path: ## from a config.pth file
                self.config = torch.load(self.fb3config.finetuned_config_path) 
            else: ## load from a model
                if FB3Config.model_path: ## from the local folder
                    self.config = AutoConfig.from_pretrained(FB3Config.model_path, 
                                                             ouput_hidden_states=True)
                else: # from the huggingface repo
                    self.config = AutoConfig.from_pretrained(FB3Config.model_name, 
                                                            ouput_hidden_states=True)
                    self.config.hidden_dropout = 0.
                    self.config.hidden_dropout_prob = 0.
                    self.config.attention_dropout = 0.
                    self.config.attention_probs_dropout_prob = 0. 
        else: 
            self.model = AutoModel.from_pretrained(FB3Config.model_name, 
                                                    config=self.config) 

        if FB3Config.pooling == 'mean':
            self.pool = MeanPooling()
        elif FB3Config.pooling == 'max':
            self.pool = MaxPooling()
        elif FB3Config.pooling == 'min':
            self.pool = MinPooling()
        elif FB3Config.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif FB3Config.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, 
                                             layer_start=FB3Config.layer_start, 
                                             layer_weights = None)        
        ## a fully-connected layer to output the 6 scores
        self.fc = nn.Linear(self.config.hidden_size, self.CFG.n_targets)
   

    def features(self, inputs):
        outputs = self.model(**inputs)
        if FB3Config.pooling != 'weightedlayer':
            last_hidden_states = outputs[0]
            features = self.pool(last_hidden_states, inputs['attention_mask'])
        else:
            all_layer_embeddings = outputs[1]
            features = self.pool(all_layer_embeddings)
        return features
    

    def forward(self, inputs):
        features = self.features(inputs)
        outout = self.fc(features)
        return outout   