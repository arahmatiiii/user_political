"""
    data loader
"""
import pandas as pd


def read_csv(path: str, columns: list = None, names: list = None) -> pd.DataFrame:
    """

    :param path:
    :param columns:
    :param names:
    :return:
    """
    dataframe = pd.read_csv(path, usecols=columns).sample(frac=1) if columns else pd.read_csv(path).sample(frac=1)
    return dataframe.rename(columns={c: n for c, n in zip(columns, names)}) if names else dataframe

