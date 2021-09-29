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
    def __init__(self, num_classes, bert_model_path, lr, max_len,
                 n_filters, filter_sizes, hidden_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.F_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.lstm_input_shape = self.lstm_input(filter_sizes, max_len)
        self.lr = lr

        self.model = BertModel.from_pretrained(bert_model_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, self.model.config.hidden_size))
            for fs in filter_sizes])

        self.lstm_r = nn.LSTM(self.lstm_input_shape,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout)

        self.lstm_l_1 = nn.LSTM(self.model.config.hidden_size,
                                hidden_size=hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout)

        self.lstm_l_2 = nn.LSTM(self.model.config.hidden_size + (2 * hidden_dim),
                                hidden_size=hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout)
        self.W_s1 = nn.Linear(2 * hidden_dim, 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(30 * 2 * hidden_dim, num_classes), )

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    @staticmethod
    def lstm_input(filter_sizes, fix_len):
        lstm_input_shape = 0
        for item in filter_sizes:
            lstm_input_shape += (fix_len - item + 1)
        return lstm_input_shape

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, batch):
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        output = self.model(input_ids=input_ids, attention_mask=att_mask)

        r_embedded = output.last_hidden_state.unsqueeze(1)
        # [bz, 1, max_len, bert_emb_dim]
        r_conved = [F.relu(conv(r_embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        r_conved = torch.cat(r_conved, dim=2)
        r_conved = r_conved.permute(1, 0, 2)
        # conved = [n_filter, batch size, sum[sent len - filter_sizes[n] + 1]]
        r_lstm_output, (_, _) = self.lstm_r(r_conved)
        # output.size() = (n_filter, batch_size, 2*hidden_size)

        l_embedded = output.last_hidden_state.permute(1, 0, 2)
        l_lstm_output_1, (_, _) = self.lstm_l_1(l_embedded)
        residual_1 = torch.cat((l_lstm_output_1, l_embedded), dim=2)
        l_lstm_output_2, (_, _) = self.lstm_l_2(residual_1)
        # output.size() = [sent len, batch size, hid dim * n directions]

        lstm_concat = torch.cat((l_lstm_output_2, r_lstm_output), dim=0)
        lstm_concat = lstm_concat.permute(1, 0, 2)
        # output.size() = [batch size, fix_len + n_filters, hid dim * n directions]

        attn_weight_matrix = self.attention_net(lstm_concat)
        # attn_weight_matrix.size() = (batch_size, r, fix_len + n_filters)

        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_concat)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)

        final_output = self.fully_connected_layers(
            hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))

        return final_output

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
