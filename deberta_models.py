import torch
from torch import nn
from transformers import AutoModel, \
                         AutoConfig

## local imports
from torch_utils import MeanPooling, MaxPooling, MinPooling, \
                        AttentionPooling, WeightedLayerPooling
from config import CustomeDebertaModelConfig, CFG, \
                   DEBERTA_FINETUNED_CONFIG_PATH, \
                   DEBERTAV3BASE_MODEL_PATH



class EssayModel(nn.Module):
    """
        deberta-v3-base (frozen) + mean-pooling + 2 fully-connected layers
    """
    def __init__(self, 
                 config, 
                 n_targets=6):
        super().__init__()

        self.model_name = config.model
        self.freeze = config.freeze_encoder

        ## freeze the original model
        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        ## attach mean-pooling and fully-connected layers
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(config.dropout) ## dropout 
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, 64) ## 1 fully connected layer is enought
        self.fc2 = nn.Linear(64, n_targets)


    def forward(self, inputs):
        outputs = self.encoder(**inputs, return_dict=True) ## kwarg expansion is not supported by torch script
        outputs = self.pooler(outputs['last_hidden_state'], inputs['attention_mask'])
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs  



class CustomDebertaModel(nn.Module):
    """
        deberta-v3-base + 1 pooling + 1 fully-connected layers
    """
    def __init__(self, 
                 config:CFG, 
                 config_path=DEBERTA_FINETUNED_CONFIG_PATH,
                 pretrained=False):
        super().__init__()
 
        self.CFG = config ## it is useless. have it only to load a custome checkpoint
        if config_path is None:
            self.config = AutoConfig.from_pretrained(config.model_path, 
                                                     ouput_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.            
        else:
            print(f"loading config from: '{config_path}'")
            self.config = torch.load(config_path)   
        
        if pretrained:
            self.model = AutoModel.from_pretrained(config.model_path, 
                                                   config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        ## add mean-pooling and fully-connected layers
        if config.pooling=='mean':
            self.pool = MeanPooling()
        elif config.pooling=='max':
            self.pool=MaxPooling()
        elif config.pooling=='min':
            self.pool = MinPooling()
        elif config.pooling=='attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif config.pooling=='weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers,
                                             layer_start=config.layer_start,
                                             layer_weights=None)
        self.fc = nn.Linear(self.model.config.hidden_size, 
                            config.n_targets)


    def forward(self, inputs, config):
        outputs = self.model(**inputs, return_dict=True) ## kwarg expansion is not supported by torch script
        if config.pooling!='weightedlayer':
            last_hidden_states = outputs[0]
            outputs = self.pooler(last_hidden_states, inputs['attention_mask'])
        else:
            all_layer_embeddings = outputs[1]
            outputs = self.pooler(all_layer_embeddings)
        outputs = self.fc(outputs)
        return outputs


class FB3Model(nn.Module):
    def __init__(self, 
                 CFG:CFG, 
                 config_path=DEBERTA_FINETUNED_CONFIG_PATH,
                 pretrained=True):
        super().__init__()

        self.CFG = CFG 
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(CFG.model_path, 
                                                     ouput_hidden_states = True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.              
        else:
            self.config = torch.load(config_path)   
        
        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model_path, 
                                                   config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
 
        if CFG.pooling == 'mean':
            self.pool = MeanPooling()
        elif CFG.pooling == 'max':
            self.pool = MaxPooling()
        elif CFG.pooling == 'min':
            self.pool = MinPooling()
        elif CFG.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif CFG.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, 
                                             layer_start = CFG.layer_start, 
                                             layer_weights = None)        
        self.fc = nn.Linear(self.config.hidden_size, self.CFG.n_targets)
   

    def feature(self,inputs):
        outputs = self.model(**inputs)
        if CFG.pooling != 'weightedlayer':
            last_hidden_states = outputs[0]
            feature = self.pool(last_hidden_states,inputs['attention_mask'])
        else:
            all_layer_embeddings = outputs[1]
            feature = self.pool(all_layer_embeddings)
        return feature
    

    def forward(self,inputs):
        feature = self.feature(inputs)
        outout = self.fc(feature)
        return outout