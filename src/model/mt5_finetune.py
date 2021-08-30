import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.functional import F

from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer, get_linear_schedule_with_warmup


class CustomDataset(Dataset):
    def __init__(self, first_cul: list, second_cul: list, targets: list, tokenizer, max_len):
        self.first_cul = first_cul
        self.second_cul = second_cul
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item_index):
        premises = self.first_cul[item_index]
        premises = " ".join(str(premises).split()[:50])

        hypotheses = self.second_cul[item_index]
        hypotheses = " ".join(str(hypotheses).split()[:40])

        label = self.targets[item_index]

        inputs_ids = self.tokenizer.encode_plus(text=premises, text_pair=hypotheses,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                return_tensors="pt",
                                                padding='max_length',
                                                truncation=True,
                                                return_token_type_ids=True).input_ids

        inputs_ids = inputs_ids.flatten()

        return {"inputs_ids": inputs_ids, "targets": torch.tensor(label)}


class DataModule(pl.LightningDataModule):

    def __init__(self, train_first_cul, train_second_cul, train_target, val_first_cul, val_second_cul,
                 val_target, test_first_cul, test_second_cul, test_target, batch_size, num_workers, tokenizer_path,
                 max_len):
        super().__init__()
        self.train_first_cul = train_first_cul
        self.train_second_cul = train_second_cul
        self.train_target = train_target

        self.val_first_cul = val_first_cul
        self.val_second_cul = val_second_cul
        self.val_target = val_target

        self.test_first_cul = test_first_cul
        self.test_second_cul = test_second_cul
        self.test_target = test_target

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        self.train_dataset = CustomDataset(first_cul=self.train_first_cul, second_cul=self.train_second_cul,
                                           targets=self.train_target, tokenizer=self.tokenizer, max_len=self.max_len)
        self.val_dataset = CustomDataset(first_cul=self.val_first_cul, second_cul=self.val_second_cul,
                                         targets=self.val_target, tokenizer=self.tokenizer, max_len=self.max_len)
        self.test_dataset = CustomDataset(first_cul=self.test_first_cul, second_cul=self.test_second_cul,
                                          targets=self.test_target, tokenizer=self.tokenizer, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Classifier(pl.LightningModule):
    def __init__(self, num_classes, t5_model_path, lr, n_filters, filter_sizes, dropout):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)

        self.lr = lr

        self.model = T5EncoderModel.from_pretrained(t5_model_path)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, 1024))
            for fs in filter_sizes
        ])
        self.classifier = nn.Linear(n_filters * len(filter_sizes), num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        inputs_ids = batch['inputs_ids']
        output_encoder = self.model(inputs_ids).last_hidden_state
        output_encoder = output_encoder.unsqueeze(1)

        conved = [F.relu(conv(output_encoder)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)

        final_output = self.classifier(cat)
        return final_output

    def training_step(self, batch, batch_idx):
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_f1_first_class', self.f_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_f1_second_class', self.f_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'predictions': outputs, 'labels': label}

    def validation_step(self, batch, batch_idx):
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1_first_class', self.f_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_f1_second_class', self.f_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        label = batch['targets'].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.accuracy(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_f1_first_class', self.f_score(torch.softmax(outputs, dim=1), label)[0],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_f1_second_class', self.f_score(torch.softmax(outputs, dim=1), label)[1],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_total_F1', self.F_score_total(torch.softmax(outputs, dim=1), label),
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
