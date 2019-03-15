import ast

from utils.im_utils import save_ims


def convert_data_type(var_string, data_type):
    """
    Convert string to relevant data type
    :param var_string: variable as a string (e.g.: '[0]', '1', '2.0', 'hellow')
    :param data_type: the type of the data
    :return: string converted to data_type
    """
    if data_type is str:
        return var_string

    if data_type is float:
        return float(var_string)

    if data_type is int:
        return int(var_string)

    if data_type is bool:
        return ast.literal_eval(var_string)

    if data_type is list:
        return ast.literal_eval(var_string)

    if data_type is tuple:
        return ast.literal_eval(var_string)


if __name__ == '__main__':
    save_ims('./test_data/9968924_V01-Aug00_056')
