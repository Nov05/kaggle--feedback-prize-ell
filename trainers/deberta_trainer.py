
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import gc
import os

## local imports
from torch_utils import EssayDataset
from trainers.base_trainers import ModelTrainer
from config import DEBERTA_FINETUNED_MODEL_PATH, \
                   TRAINING_PARAMS, \
                   CustomeDebertaModelConfig, CFG
from deberta_models import EssayModel, \
                           CustomDebertaModel, FB3Model



class DebertaTrainer(ModelTrainer):
    def __init__(self, 
                 model=None, ## fine-tuned model object
                 model_path=DEBERTA_FINETUNED_MODEL_PATH, ## path to fine-tuned model .pth
                #  config=type('EssayModelConfig', (), TRAINING_PARAMS["deberta"]), ## convert dict to obj
                 config=CFG, 
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

        self.model_path = model_path ## fine-tuned model
        self.config = config 
        self.accelerator = accelerator if accelerator else self._get_accelerator()
        self.model = model if model else self.get_model(self.config, self.model_path, self.accelerator)
        self.tokenizer = tokenizer if tokenizer else self.get_tokenizer(config.model_path) ## deberta-v3-base
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
        # model = EssayModel(config)
        # model = CustomDebertaModel(config)
        model = FB3Model(config)
        model_checkpoint = torch.load(model_path,
                                      map_location=accelerator.device)
        print(f"loading model state dict from: '{model_path}'")
        model.load_state_dict(model_checkpoint['model'], strict=False)
        return model


    @staticmethod
    def get_tokenizer(model_path): 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer


    @staticmethod
    def class_to_dict():
        pass


    def _get_datasets(self, file_name=None, is_test=False):
        df = self.load_data(file_name, is_test=is_test)
        if not is_test:
            df_train, df_val = self.split_data(df, test_size=self.config.test_size)
            train_dataset = EssayDataset(df_train, self.config, tokenizer=self.tokenizer, is_test=is_test)
            val_dataset = EssayDataset(df_val, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return train_dataset, val_dataset
        else:
            test_dataset = EssayDataset(df, self.config, tokenizer=self.tokenizer, is_test=is_test)
            return test_dataset
    

    def _get_data_loader(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.config.batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  pin_memory=True)
        return data_loader


    def _get_accelerator(self):
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        return accelerator
    

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=self.config.adam_eps)
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
    def eval_one_epoch(self, epoch):
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
    def test(self, test_loader=None, recast_scores=True, write_file=True):
        self.model.eval()
        if not test_loader:
            print(f"loading test data from: '{self._test_file_name}'")
            test_dataset = self._get_datasets(is_test=True)
            test_loader = self._get_data_loader(test_dataset)

        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        print(f"inference device: {self.accelerator.device}")

        test_progress = tqdm(test_loader, total=len(test_loader), desc="testing...")
        preds = []
        for (inputs) in test_progress:
            with torch.no_grad():
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