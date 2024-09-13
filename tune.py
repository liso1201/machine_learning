import warnings
warnings.filterwarnings('ignore')

from random import randint
random_seed=randint(1, 10**6 - 1)

import numpy as np

from sklearn.model_selection import train_test_split

from typology import (
    ScoreFunction, ScoreLike, 
    Scalar, ScalarOrIterable, 
    Hiperparams, HiperparamGrid, 
    SklearnModel, SKlearnReescaler)

from typing import Optional, Callable, Mapping
from collections.abc import Iterable
from pandas import DataFrame, Series

from model_utiles import get_attributes
from utiles import kwargs_parser

from search.methods import bayesian_search, intensive_search
from artifact import Artifact

from score_utiles import provide_scores

def adapt_featureValue(v: ScalarOrIterable) -> np.ndarray:
    """
    Convert a scalar or iterable to a one-dimensional NumPy array.

    Parameters
    ----------
    v : Scalar or Iterable
        A scalar value or an iterable of scalar values. Scalars will be converted to a single-element array,
        and iterables will be converted to a one-dimensional array with all elements.

    Returns
    -------
    np.ndarray
        A one-dimensional NumPy array containing the value(s) from the input. Scalars are converted to 
        a single-element array, while iterables are flattened into a one-dimensional array.

    Raises
    ------
    ValueError
        If the input is neither a scalar nor an iterable, a ValueError is raised with an appropriate message.

    Notes
    -----
    - Scalars are first converted to a NumPy array with `np.atleast_1d` and then flattened using `ravel()`.
    - Iterables are first converted to a tuple and then to a NumPy array, followed by flattening with `ravel()`.
    """
    if isinstance(v, Scalar):
        return np.atleast_1d(v).ravel()
    elif isinstance(v, Iterable):
        return np.array(tuple(v)).ravel()
    else:
        raise ValueError(f'Expected a scalar or iterable, but got {type(v).__name__}.')

def adapt_hiperparams(hiperparamsLike: Mapping[str, ScalarOrIterable]) -> HiperparamGrid:
    """
    Convert a mapping of hyperparameters, where values can be scalars or iterables, to a dictionary of NumPy arrays.

    Parameters
    ----------
    hiperparamsLike : Mapping[str, ScalarOrIterable]
        A dictionary where keys are parameter names and values are either scalars or iterables of scalar values.

    Returns
    -------
    HiperparamGrid
        A dictionary with the same keys as the input but with values converted to one-dimensional NumPy arrays.

    Notes
    -----
    - Each value in the input dictionary is processed using `adapt_featureValue` to ensure it is converted 
      to a one-dimensional NumPy array.
    """
    new = {}

    for k, v in hiperparamsLike.items():
        new[k] = adapt_featureValue(v)

    return new

# =============
# MAIN FUNCTION
# =============
def training(
    data: DataFrame,
    variables: Iterable[str],
    target: str,
    preditor: SklearnModel,
    params: Mapping[str, ScalarOrIterable],
    test_data: Optional[DataFrame]=None,
    scaler: SKlearnReescaler=None,
    bayesian: bool=False,
    n_iters: Optional[int] = None,
    scores: Optional[ScoreLike] = None,
    lesser_is_better: bool = True,
    dirname: Optional[str] = None,
    **kwargs) -> Optional[Artifact]:
    
    if test_data is None:
        kw = kwargs_parser(train_test_split, kwargs, ignore=('arrays') )
        train_data, test_data = train_test_split(data, **kw)
    else:
        train_data = data.copy()
    
    xtrain: DataFrame = train_data.get(variables)
    ytrain: Series = train_data.get(target)

    xtest: DataFrame = test_data.get(variables)
    ytest: Series = test_data.get(target)

    # Preprocessing
    if scaler:
        scaler.fit(X=xtrain)
        xtrain.loc[::] = scaler.transform(xtrain)   # still dataframe
        xtest.loc[::] = scaler.transform(xtest)     # still dataframe
        
    # Handle random_state in hiperparams
    if hasattr(preditor, 'random_state') and 'random_state' not in params:
        params['random_state'] = [random_seed]

    # verify hiperparameters
    space: HiperparamGrid = adapt_hiperparams(params)
    
    # Initialize scores or return defaults if not provided
    scores: dict[str, ScoreFunction] = provide_scores(scores, preditor)
    
    fsearch: Callable = bayesian_search if bayesian else intensive_search
    kw = kwargs_parser(fsearch, kwargs, ignore=locals(), restricted=False)

    bestModel, hiperparams_search = fsearch(
        hiperparams=space, 
        preditor=preditor, 
        xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest, 
        scores=scores, lesser_is_better=lesser_is_better, n_iters=n_iters, **kw)         
    
    # Creating Artifact
    hiperparams_of_bestModel: Hiperparams = get_attributes(bestModel, *params)

    artifact = Artifact(
        data=data,
        variables=variables,
        target=target,
        preditor=bestModel,
        hiperparams=hiperparams_of_bestModel,
        scaler=scaler,
        test_data=test_data,
        train_data=train_data,
        n_iters=n_iters,
        hiperparams_search=hiperparams_search
    )

    if dirname:
        return artifact.to_shelve(dirname=dirname)
    
    return artifact

if __name__ == '__main__':
    pass



