from typing import List


def build_exact_match(source_data: List[list],
                      compare_data: List[list],
                      skip_item=None) -> list:
    """
    if item from source_data exist in compare_data append 2 to exact_match list
    if item not exact_match in compare_data append 3 to exact_match list
    if item equal to  skip_item append 1 to exact_match list
    :param source_data:
    :param compare_data:
    :param skip_item:
    :return:
    """
    output_data = list()
    for data_1, data_2 in zip(source_data, compare_data):
        exact_match = list()
        for item in data_1:
            if skip_item and (item == skip_item):
                exact_match.append(1)
            elif item in data_2:
                exact_match.append(2)
            else:
                exact_match.append(3)
        output_data.append(exact_match)
    return output_data
