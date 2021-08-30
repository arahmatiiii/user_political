import torch
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint

from data_preparation import token_padding, characters_padding


def build_checkpoint_callback(save_top_k, filename='QTag-{epoch:02d}-{val_loss:.2f}', monitor='val_loss'):
    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,  # monitored quantity
        filename=filename,
        save_top_k=save_top_k,  # save the top k models
        mode='min',  # mode of the monitored quantity for optimization

    )
    return checkpoint_callback


def pad_collator(batch, pad_idx):
    batch_temp = dict()
    batch_keys = batch[0].keys()
    for item in batch_keys:
        if 'target' in item:
            padded_temp = torch.tensor(list(itertools.chain(*[batch[i][item] for i in range(len(batch))])))
        elif 'char' in item:
            temp = [batch[i][item] for i in range(len(batch))]
            padded_temp = characters_padding(temp, pad_index=pad_idx)
            padded_temp = [torch.tensor(sample) for sample in padded_temp]
        else:
            temp = [batch[i][item] for i in range(len(batch))]
            padded_temp = torch.tensor(token_padding(temp, pad_index=pad_idx))

        batch_temp[item] = padded_temp
    return batch_temp
