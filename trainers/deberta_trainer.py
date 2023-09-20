
import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

## local imports
from torch_utils import EssayDataset
from trainers.base_trainers import ModelTrainer
from config import TEST_SIZE, DEBERTAV3BASE_MODEL_PATH
from deberta_models import EssayModel, FB3Model



class DebertaTrainer(ModelTrainer):
    def __init__(self, 
                 model_type=None, ## e.g. 'deberta2'
                 model=None, ## fine-tuned model object
                 model_path=None, ## path to fine-tuned model .pth, e.g. DEBERTA_FINETUNED_MODEL_PATH2
                 config=None, 
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
        assert config, "provide config!"

        self.model_type = model_type
        self.model_path = model_path ## fine-tuned model
        self.config = config 
        self.is_test = is_test
        self.accelerator = accelerator if accelerator else self._get_accelerator()
        self.model = model if model else self.get_model(self.model_type, self.config, self.model_path, self.accelerator)
        self.tokenizer = tokenizer if tokenizer else self.get_tokenizer(DEBERTAV3BASE_MODEL_PATH) ## deberta-v3-base
        self.input_keys = ['input_ids', 'token_type_ids', 'attention_mask']

        ## if a saved model is loaded for inference only, there won't be train-val loaders etc.
        if not is_test:
            self.train_dataset, self.val_dataset = self._get_datasets()
            self.train_loader = self._get_data_loader(self.train_dataset)
            self.val_loader = self._get_data_loader(self.val_dataset)
            self.optim = self._get_optimizer(self.model_type)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optim,
                T_0=5,
                eta_min=1e-7
            )
            self.train_losses, self.val_losses = [], []


    @staticmethod
    def get_model(model_type, config, model_path, accelerator):
        print(f"loading model state dict from: '{model_path}'")
        if model_type=='deberta1':
            model = EssayModel(config)
            if accelerator:
                model.load_state_dict(torch.load(model_path, map_location=accelerator.device))
            else:
                model.load_state_dict(torch.load(model_path))
        elif model_type=='deberta2':
            # model = CustomDebertaModel(config)
            model = FB3Model(config)
            if accelerator:
                model_checkpoint = torch.load(model_path,
                                              map_location=accelerator.device)
            else:
                model_checkpoint = torch.load(model_path)
            model.load_state_dict(model_checkpoint['model'], strict=False)
        return model


    @staticmethod
    def get_tokenizer(model_path): 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer


    @staticmethod
    def split_data(df, test_size, random_state=42):
        df_train, df_test = train_test_split(df, 
                                             test_size=test_size, 
                                             random_state=random_state)
        return df_train, df_test


    def _get_datasets(self, file_name=None, is_test=False):
        df = self.load_data(file_name, is_test=is_test)
        if is_test:
            test_dataset = EssayDataset(df, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return test_dataset
        else:
            df_train, df_val = self.split_data(df, test_size=TEST_SIZE)
            train_dataset = EssayDataset(df_train, self.config, tokenizer=self.tokenizer, is_test=is_test)
            val_dataset = EssayDataset(df_val, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return train_dataset, val_dataset
    

    def _get_data_loader(self, dataset, is_test=False):
        ## CAUTION: don't shuffle test data. or you will mess the test score
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.config.batch_size,
                                                  shuffle=(not is_test), 
                                                  num_workers=2,
                                                  pin_memory=True)
        return data_loader


    def _get_accelerator(self):
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        return accelerator
    

    def _get_optimizer_grouped_parameters(self,
                                          model, 
                                          layerwise_lr,
                                          layerwise_weight_decay,
                                          layerwise_lr_decay):

        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = \
            [{"params": [p for n,p in model.named_parameters() if "model" not in n],
              "weight_decay": 0.0,
              "lr": layerwise_lr}]
        # initialize lrs for every layer of deberta-v3-base
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += \
                [{"params": [p for n,p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                  "weight_decay": layerwise_weight_decay,"lr": lr,},
                 {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0,"lr": lr,},]
            lr *= layerwise_lr_decay
        return optimizer_grouped_parameters


    def _get_optimizer(self, model_type):
        if model_type=='deberta1':
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n,p in self.model.encoder.base_model.named_parameters() 
                            if not any(nd in n for nd in no_decay)], 
                 'weight_decay': 0.01},
                {'params': [p for n,p in self.model.encoder.base_model.named_parameters() 
                            if any(nd in n for nd in no_decay)], 
                 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                          lr=self.config.lr, 
                                          eps=self.config.adam_eps)
        elif model_type=='deberta2':
            grouped_optimizer_params = self._get_optimizer_grouped_parameters(
                self.model, 
                self.config.layerwise_lr,
                self.config.layerwise_weight_decay,
                self.config.layerwise_lr_decay
                )
            optimizer = AdamW(grouped_optimizer_params,
                            lr=self.config.layerwise_lr,
                            eps=self.config.layerwise_adam_epsilon)
        return optimizer
    

    def _accelerator_prepare(self):
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
        self.model.train()
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
        self.clear()


    @torch.no_grad()
    def eval_one_epoch(self):
        self.model.eval()
        running_loss = 0.
        eval_progress = tqdm(self.val_loader, total=len(self.val_loader), desc="evaluating...")
        for (inputs,targets) in eval_progress:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets['labels'])
            running_loss += loss.item()
            del inputs, targets, outputs, loss
        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        self.clear()


    def train(self):
        self._accelerator_prepare()
        train_progress = tqdm(range(1, self.config['epochs']+1),
                              leave=True,
                              desc="training...")
        for epoch in train_progress:
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.eval_one_epoch(epoch)
            print(f"{'='*10} EPOCH {epoch} / {self.config.num_epochs} {'='*10}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"val loss: {self.val_losses[-1]}\n\n")


    @torch.no_grad()
    def test(self, 
             data_loader=None, 
             recast_scores=True, 
             write_submission_file=True):

        if not data_loader:
            print(f"loading test data from: '{self._test_file_name}'")
            test_dataset = self._get_datasets(is_test=True)
            data_loader = self._get_data_loader(test_dataset, is_test=True)

        self.model, data_loader = self.accelerator.prepare(self.model, data_loader)
        print(f"inference device: {self.accelerator.device}")

        self.model.eval()
        test_progress = tqdm(data_loader, total=len(data_loader), desc="inferring...")
        preds = []
        for (inputs,_) in test_progress:
            # with torch.no_grad():
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
        preds = torch.concat(preds).numpy()     
        self.clear()

        if recast_scores:
            print(f"recasting scores...")
            preds = self.recast_scores(preds)
        else:
            print(f"no recasting scores")

        if write_submission_file:
            submission_df = super().load_data(is_test=True)
            super().make_submission_file(submission_df, preds)
        return preds


    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()