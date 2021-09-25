import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.functional import F

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


class CustomDataset(Dataset):
    def __init__(self, first_cul: list, targets: list, tokenizer, max_len):
        self.first_cul = first_cul
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        input_text = self.first_cul[item_index]
        label = self.targets[item_index]

        inputs = self.tokenizer.encode_plus(text=input_text,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_tensors="pt",
                                            padding='max_length',
                                            truncation=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=True)

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        return {
            'input_ids': input_ids, 'attention_mask': attn_mask,
            'target': torch.tensor(label)
        }


class DataModule(pl.LightningDataModule):

    def __init__(self, train_first_cul, train_target, val_first_cul, val_target, test_first_cul, test_target,
                 batch_size, num_workers, tokenizer_path,
                 max_len):
        super().__init__()
        self.train_first_cul = train_first_cul
        self.train_target = train_target

        self.val_first_cul = val_first_cul
        self.val_target = val_target

        self.test_first_cul = test_first_cul
        self.test_target = test_target

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        self.train_dataset = CustomDataset(first_cul=self.train_first_cul, targets=self.train_target,
                                           tokenizer=self.tokenizer, max_len=self.max_len)
        self.val_dataset = CustomDataset(first_cul=self.val_first_cul, targets=self.val_target,
                                         tokenizer=self.tokenizer, max_len=self.max_len)
        self.test_dataset = CustomDataset(first_cul=self.test_first_cul, targets=self.test_target,
                                          tokenizer=self.tokenizer, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Classifier(pl.LightningModule):
    def __init__(self, num_classes, bert_model_path, max_len, lr):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.F_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)

        self.lr = lr

        self.model = BertModel.from_pretrained(bert_model_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        output = self.model(input_ids=input_ids, attention_mask=att_mask)
        output = self.classifier(output.pooler_output)
        return output

    def training_step(self, batch, batch_idx):
        label = batch['target'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_f1_first_class', self.F_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_f1_second_class', self.F_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'predictions': outputs, 'labels': label}

    def validation_step(self, batch, batch_idx):
        label = batch['target'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1_first_class', self.F_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1_second_class', self.F_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        label = batch['target'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_f1_first_class', self.F_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_f1_second_class', self.F_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
