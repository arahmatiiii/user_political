from itertools import chain
from collections import Counter
from typing import List


def item_counter(items: List[list]) -> dict:
    """

    :param items: ex: [["first", "sent"], ["second", "sent"]]
    :return: {"first": 1, "sent":2, "second": 1}
    """
    counter = Counter()
    for item in items:
        counter.update(item)

    return dict(counter)


def convert_words_to_chars(tokens: list) -> list:
    """

    :param tokens:
    :return:
    """
    return list(set(chain.from_iterable(tokens)))


def find_max_length_in_list(data: List[list]) -> int:
    """

    :param data:
    :return:
    """
    return max(len(sample) for sample in data)
