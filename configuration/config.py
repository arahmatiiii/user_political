import argparse
import math
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw/")

        self.parser.add_argument("--embedding_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Embeddings/")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--t5_model_en_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/t5_model_en_large")

        self.parser.add_argument("--t5_tokenizer_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/t5_tokenizer_large")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--train_file", type=str, default="scitail/scitail_1.0_train.csv")
        self.parser.add_argument("--test_file", type=str, default="scitail/scitail_1.0_test.csv")
        self.parser.add_argument("--dev_file", type=str, default="scitail/scitail_1.0_dev.csv")

        self.parser.add_argument("--embedding_file", type=str, default="skipgram_news_300d_30e.txt")

        self.parser.add_argument("--data_headers", type=list, default=["premises", "hypotheses", "targets"])
        self.parser.add_argument("--customized_headers", type=list, default=["first_text", "second_text", "targets"])

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_filters", type=int,
                                 default=64,
                                 help="...")

        self.parser.add_argument("--filter_sizes", type=list,
                                 default=[3, 4, 5],
                                 help="...")

        self.parser.add_argument("--dropout", type=float,
                                 default=0.25,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=32,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--max_sentence_len", type=int,
                                 default=100)

    def get_config(self):
        return self.parser.parse_args()
