import ast
from typing import Sequence


def convert_data_type(var_string, original):
    """
    Convert string to relevant data type
    :param var_string: variable as a string (e.g.: '[0]', '1', '2.0', 'hellow')
    :param original: original variable
    :return: string converted to data_type
    """
    if isinstance(original, str):
        return var_string

    if isinstance(original, float):
        return float(var_string)

    if isinstance(original, int):
        return int(var_string)

    if isinstance(original, bool):
        return ast.literal_eval(var_string)

    if isinstance(original, Sequence) or isinstance(original, type(None)):
        return ast.literal_eval(var_string)
