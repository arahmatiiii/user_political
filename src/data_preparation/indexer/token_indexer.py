from typing import List
from .indexer import Indexer


class TokenIndexer(Indexer):
    def __init__(self, vocabs: list, pad_index: int = 0, unk_index: int = 1):
        super().__init__(vocabs)
        self.pad_index = pad_index
        self.unk_index = unk_index

    def get_idx(self, word):
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if word in self._vocab2idx.keys():
            return self._vocab2idx[word]
        return self._vocab2idx["<UNK>"]

    def get_word(self, idx):
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab.keys():
            return self._idx2vocab[idx]
        return self._idx2vocab[self.unk_index]

    def build_vocab2idx(self):
        """

        :return:
        """
        # TODO: add custom pad_index and unk_index
        self._vocab2idx = dict()
        self._vocab2idx["<PAD>"] = self.pad_index
        self._vocab2idx["<UNK>"] = self.unk_index
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    def build_idx2vocab(self):
        """

        :return:
        """
        # TODO: add custom pad_index and unk_index
        self._idx2vocab = dict()
        self._idx2vocab[self.pad_index] = "<PAD>"
        self._idx2vocab[self.unk_index] = "<UNK>"
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def convert_samples_to_char_indexes(self, tokenized_samples: List[list]) -> List[list]:
        """
        :param tokenized_samples: [[word_1, ..., word_n], ..., [word_1, ..., word_m]]
        :return: [[[ch1, ch2, ..], ..., [ch1, ch2, ..]], ..., [[ch1, ch2, ..], ..., [ch1, ch2, ..]]]
        """
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                chars = list()
                for char_index, char in enumerate(str(token)):
                    chars.append(self.get_idx(char))
                tokenized_samples[index][token_index] = chars
        return tokenized_samples
