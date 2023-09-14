
import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import gc

## local imports
from trainers.base_trainers import ModelTrainer
from torch_utils import EssayDataset, MeanPooling
from config import TEST_SIZE, TRAINING_PARAMS, \
                   DEBERTA_FINETUNED_MODEL_PATH

    

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



class DebertaTrainer(ModelTrainer):
    def __init__(self, 
                 model=None, ## fine-tuned model object
                 model_path=DEBERTA_FINETUNED_MODEL_PATH, ## path to fine-tuned model .pth
                 config=TRAINING_PARAMS["debertav3base"], ## config dict
                 tokenizer=None,
                 accelerator=None,
                 target_columns=None,
				 feature_columns=None,
				 train_file_name=None,
				 test_file_name=None,
				 submission_file_name=None,
                 is_test=False):
        super().__init__(target_columns=target_columns,
						 feature_columns=feature_columns,
						 train_file_name=train_file_name,
						 test_file_name=test_file_name,
						 submission_file_name=submission_file_name)
        self.model_path = model_path
        self.config = config 
        self.accelerator = accelerator if accelerator else self._get_accelerator()
        self.model = model if model else self.get_model(self.config, self.model_path, self.accelerator)
        self.tokenizer = tokenizer if tokenizer else self.get_tokenizer(self.config['model']) ## deberta-v3-base
        self.input_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        self.is_test = is_test

        ## if a saved model is loaded for inference only, there won't be train-val loaders etc.
        if not is_test:
            self.train_dataset, self.val_dataset = self._get_datasets()
            self.train_loader = self._get_data_loader(self.train_dataset)
            self.val_loader = self._get_data_loader(self.val_dataset)
            self.optim = self._get_optimizer()
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optim,
                T_0=5,
                eta_min=1e-7
            )
            self.train_losses, self.val_losses = [], []


    @staticmethod
    def get_model(config, model_path, accelerator):
        print(f"loading model from: '{model_path}'")
        model = EssayModel(config)
        model.load_state_dict(torch.load(model_path, map_location=accelerator.device))
        return model


    @staticmethod
    def get_tokenizer(model_path): 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer


    def _get_datasets(self, file_name=None, is_test=False):
        df = self.load_data(file_name, is_test=is_test)
        if not is_test:
            df_train, df_val = self.split_data(df, test_size=TEST_SIZE)
            train_dataset = EssayDataset(df_train, self.config, tokenizer=self.tokenizer, is_test=is_test)
            val_dataset = EssayDataset(df_val, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return train_dataset, val_dataset
        else:
            test_dataset = EssayDataset(df, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return test_dataset
    

    def _get_data_loader(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True,
                                                  num_workers=2,
                                                  pin_memory=True)
        return data_loader


    def _get_accelerator(self):
        accelerator = Accelerator(gradient_accumulation_steps=self.config['gradient_accumulation_steps'])
        return accelerator
    

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['adam_eps'])
        return optimizer
    

    def accelerator_prepare(self):
        self. model, self.optim, self.train_loader, self.val_loader, self.scheduler \
            = self.accelerator.prepare(self.model,
                                       self.optim,
                                       self.train_loader,
                                       self.val_loader,
                                       self.scheduler)


    def loss_fn(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss


    def train_one_epoch(self, epoch):
        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))
        for idx,(inputs,targets) in enumerate(progress):
            with self.accelerator.accumulate(self.model):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets['labels'])
                running_loss += loss.item()
                self.accelerator.backward(loss)
                self.optim.step()
                if self.config['enable_scheduler']:
                    self.scheduler.step(epoch - 1 + idx/len(self.train_loader))
                self.optim.zero_grad()
                del inputs, targets, outputs, loss
        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)


    @torch.no_grad()
    def eval_one_epoch(self, epoch):
        running_loss = 0.
        eval_progress = tqdm(self.val_loader, total=len(self.val_loader), desc="evaluating...")
        for (inputs,targets) in eval_progress:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets['labels'])
            running_loss += loss.item()
            del inputs, targets, outputs, loss
        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)


    def train(self):
        self.accelerator_prepare()
        train_progress = tqdm(range(1, self.config['epochs']+1),
                              leave=True,
                              desc="training...")
        for epoch in train_progress:
            self.model.train()
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()
            self.model.eval()
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.eval_one_epoch(epoch)
            self.clear()
            print(f"{'='*10} EPOCH {epoch} / {self.config['epochs']} {'='*10}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"val loss: {self.val_losses[-1]}\n\n")


    def test(self, test_loader=None, recast_scores=True, write_file=True):
        if not test_loader:
            print(f"loading test data from: '{self._test_file_name}'")
            test_dataset = self._get_datasets(is_test=True)
            test_loader = self._get_data_loader(test_dataset)

        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        print(f"inference device: {self.accelerator.device}")

        test_progress = tqdm(test_loader, total=len(test_loader), desc="testing...")
        preds = []
        for (inputs) in test_progress:
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
        preds = torch.concat(preds).numpy()
        self.clear()
        if recast_scores:
            print(f"recasting scores...")
            preds = self.recast_scores(preds)
        if write_file:
            submission_df = super().load_data(is_test=True)
            super().make_submission_file(submission_df, preds)
        return preds


    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()