from typing import List

from utils import find_max_length_in_list


def token_padding(data: List[list], pad_index: int) -> List[list]:
    """

    :param data:
    :param pad_index:
    :return:
    """
    max_len = find_max_length_in_list(data)
    data = [sample + [pad_index] * (max_len - len(sample)) for sample in data]
    return data


def characters_padding(data: List[list], pad_index: int) -> list:
    """

    :param data:
    :param pad_index:
    :return:
    """
    batch = list()
    sentences_max_len = find_max_length_in_list(data)
    for sentence in data:
        words_max_len = find_max_length_in_list(sentence)
        padded_words = [word + [pad_index] * (words_max_len - len(word))
                        for word in sentence]
        padded_words = padded_words + [[pad_index] * words_max_len] * \
                       (sentences_max_len - len(sentence))
        batch.append(padded_words)
    return batch
