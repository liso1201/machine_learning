import numpy as np

from pandas import DataFrame
from random import randint

from typing import Union

def split(data: DataFrame, test_size: float) -> tuple[DataFrame, DataFrame]:
    if not (0 < test_size < 1):
        raise ValueError("test_size deve ser um valor entre 0 e 1 (não inclusos)")
    
    # Embaralhar os dados para garantir aleatoriedade
    df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calcular o número de linhas para o conjunto de teste
    test_length = int(len(data) * test_size)
    
    # Fatiar o dataframe
    test_data = df_shuffled.iloc[:test_length]
    train_data = df_shuffled.iloc[test_length:]
    
    return train_data, test_data

def random_dataframe(n_vars: int=None, lenght: int=None) -> DataFrame:

    if n_vars is None:
        n_vars = randint(1, 5)
    
    if lenght is None:
        lenght = randint(100, 10000)
    
    columns = [f'var{i}' for i in range(n_vars)] + ['target']
    data = np.random.rand(lenght, n_vars + 1)

    return DataFrame(data, columns=columns)

def generate_randomData_for_tests(n_vars:int=None, lenght: int=None, test_size: float=None) -> Union[DataFrame, tuple[DataFrame, DataFrame]]:

    df = random_dataframe(n_vars, lenght)

    if test_size:
        return split(df, test_size)

    return df


if __name__ == '__main__':
    train, test = generate_randomData_for_tests(0.2)
    print(train, test, sep='\n\n')