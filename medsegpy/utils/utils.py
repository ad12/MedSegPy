import ast

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

    if data_type is list or data_type is tuple or data_type is type(None):
        return ast.literal_eval(var_string)
