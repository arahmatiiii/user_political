import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from configuration import BaseConfig
from data_loader import read_csv, write_json
from data_preparation import Indexer
from model import DataModule, Classifier, build_checkpoint_callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # load raw data
    TRAIN_DATA = read_csv(path=ARGS.train_data_path, columns=ARGS.data_headers,
                          names=ARGS.customized_headers)

    TEST_DATA = read_csv(path=ARGS.test_data_path, columns=ARGS.data_headers,
                         names=ARGS.customized_headers)

    VALID_DATA = read_csv(path=ARGS.dev_data_path, columns=ARGS.data_headers,
                          names=ARGS.customized_headers)

    TARGET_INDEXER = Indexer(vocabs=list(TRAIN_DATA.targets))
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.targets)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.targets)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    VALID_TARGETS_CONVENTIONAL = [[target] for target in list(VALID_DATA.targets)]
    VALID_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VALID_TARGETS_CONVENTIONAL)

    # Create Data Module
    DATA_MODULE = DataModule(train_first_cul=list(TRAIN_DATA.first_text),
                             train_target=TRAIN_INDEXED_TARGET,

                             val_first_cul=list(VALID_DATA.first_text),
                             val_target=VALID_INDEXED_TARGET,

                             test_first_cul=list(TEST_DATA.first_text),
                             test_target=TEST_INDEXED_TARGET,

                             batch_size=ARGS.batch_size, num_workers=ARGS.num_workers,
                             max_len=ARGS.max_sentence_len,
                             tokenizer_path=ARGS.bert_path)

    DATA_MODULE.setup()
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=ARGS.save_top_k)

    LOGGER = CSVLogger(ARGS.csv_logger_path, name="metrics_logs")

    # Instantiate the Model Trainer
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=3)
    TRAINER = pl.Trainer(max_epochs=ARGS.n_epochs, gpus=1,
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)

    # Create Model
    MODEL = Classifier(num_classes=len(set(list(TRAIN_DATA.targets))),
                       bert_model_path=ARGS.bert_path,
                       lr=ARGS.lr, max_len=ARGS.max_sentence_len)

    # Train and Test Model
    TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    TRAINER.test(ckpt_path='best', datamodule=DATA_MODULE)

    # save best mt5_model_en path
    write_json(path=ARGS.csv_logger_path + '/logs/b_model_path.json',
               data={'best_model_path': CHECKPOINT_CALLBACK.best_model_path})
