import argparse
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--t5_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/pretrained_models/..")

        self.parser.add_argument("--bert_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/pretrained_models/parsbert_v3")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--train_data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/data/Processed/train_data_normed.csv")

        self.parser.add_argument("--test_data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/data/Processed/test_data_normed.csv")

        self.parser.add_argument("--dev_data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/data/Processed/valid_data_normed.csv")

        self.parser.add_argument("--embedding_file_path", type=str,
                                 default=Path(__file__).parents[2].__str__() +
                                         "/data/Embeddings/skipgram_news_300d_30e.txt")

        self.parser.add_argument("--data_headers", type=list, default=["text", "label"])
        self.parser.add_argument("--customized_headers", type=list, default=["first_text", "targets"])

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=20,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=32,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--max_sentence_len", type=int,
                                 default=100)

        self.parser.add_argument("--filter_sizes", type=list,
                                 default=[3, 4, 5])
        self.parser.add_argument("--n_filters", type=int,
                                 default=120)
        self.parser.add_argument("--hidden_dim", type=int,
                                 default=128)
        self.parser.add_argument("--n_layers", type=int,
                                 default=1)
        self.parser.add_argument("--bidirectional", type=bool,
                                 default=True)
        self.parser.add_argument("--dropout", type=float,
                                 default=0.2)

    def get_config(self):
        return self.parser.parse_args()
