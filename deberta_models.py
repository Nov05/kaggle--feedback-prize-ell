from torch import nn
from transformers import AutoModel
from torch_utils import MeanPooling



class EssayModel(nn.Module):
    """
        EssayModel = deberta-v3-base (frozen) + mean-pooling + 2 fully-connected layers
    """
    def __init__(self, config:dict, num_classes=6):
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
        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.encoder.config.hidden_size,64)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, inputs):
        outputs = self.encoder(**inputs, return_dict=True) ## kwarg expansion is not supported by torch script
        outputs = self.pooler(outputs['last_hidden_state'], inputs['attention_mask'])
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs