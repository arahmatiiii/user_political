import numpy
import math
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TopResult:
    token: str
    embedding: numpy.array
    distance: float


class Embed:
    def __init__(self, path: str):
        self.embedding_dim: int = -1
        self.embedding_matrix: numpy.array = None
        self.token2embedding = defaultdict(numpy.array)
        self.load_embedding(path)

    def get_token_embed(self, token: str) -> numpy.array:
        if self._is_token_in_vocab(token):
            return self.token2embedding[token]
        return None

    def get_topk_tokens_by_embedding(self, token: str, k: int) -> List[TopResult]:
        results = list()
        if self._is_token_in_vocab(token):
            for item in self.token2embedding.keys():
                results.append((item, self.get_token_pair_distance(token, item)))
        results = sorted(results, key=lambda tup: tup[1], reverse=True)
        return [TopResult(token=results[top][0], distance=results[top][1],
                          embedding=self.token2embedding[results[top][0]])
                for top in range(k)]

    def get_token_pair_distance(self, first_token: str, second_token: str) -> float:
        if self._is_token_in_vocab(first_token) and self._is_token_in_vocab(second_token):
            return cosine_similarity([self.token2embedding[first_token]],
                                     [self.token2embedding[second_token]])[0][0]
        return -math.inf

    def get_token_by_embedding(self, embedding: numpy.array) -> str:
        for key, value in self.token2embedding.items():
            if (embedding == value).all():
                return key
        return ""

    def get_num_embeddings(self) -> int:
        return len(self.token2embedding)

    def get_vocabs(self) -> list:
        return list(self.token2embedding.keys())

    def build_embedding_matrix(self, word2index: dict) -> None:
        self.embedding_matrix = numpy.random.rand(len(word2index), self.embedding_dim)
        existed_tokens = set(list(word2index.keys())).intersection(list(self.token2embedding.keys()))
        for token in existed_tokens:
            token_id = word2index[token]
            self.embedding_matrix[token_id] = self.token2embedding[token]

    def get_embedding_matrix(self) -> numpy.array:
        return self.embedding_matrix

    def _is_token_in_vocab(self, token: str) -> bool:
        if token in self.token2embedding.keys():
            return True
        return False

    def load_embedding(self, path: str):
        """
        Expected file format:
            it should be a .txt file format.
            each sample also should separated by a new line.
            the tokens and embeddings should be separated by a single space.
            first line should be: num_tokens embedding_dim
            Ex: apple 0.271 -0.981 ... 0.014
        :param path:
        :return:
        """
        with open(path, mode="r", encoding="utf-8") as file:
            i = 0
            for sample in file:
                if len(sample.split()) < 301:
                    continue
                assert len(sample.split()) == 301, "bad embedding"
                i += 1
                if i < 1000:
                    token = sample.split()[0]
                    embedding = [float(item) for item in sample.split()[1:]]
                    self.token2embedding[token] = numpy.array(embedding)
                else:
                    break
        self.embedding_dim = len(list(self.token2embedding.values())[0])
