import sys
sys.path.append('.')

import numpy as np

from pandas import DataFrame, Series
from collections.abc import Generator, Iterable, Mapping
from typing import Optional, Any
from typology import SklearnModel, ScoreFunction, HiperparamGrid, Hiperparams, Scalar

from ui_tools import progress_bar

from model_utiles import create_model, evaluate_model
from utiles import is_better_than, dictK, dictzip

from itertools import product
from math import prod

from sklearn.model_selection import ParameterSampler   #ver isso depois

def searchSpace(*sequences: Iterable, n: int=None) -> Generator[Any, None, None]:
    """
    Generates combinations of elements from provided sequences.

    Parameters
    ----------
    *sequences : Iterable
        Sequences of elements from which combinations will be generated.
    n : int, optional
        Number of samples to randomly select from each sequence if provided.

    Returns
    -------
    Generator[Any, None, None]
        A generator that yields combinations of elements from the provided sequences. If `n` is specified, yields random samples from each sequence.
    
    Notes
    -----
    - If `n` is not specified, the Cartesian product of all sequences is generated (at your own risk).
    - If `n` is specified, the generator produces `n` random samples from each sequence.
    """
    return product(*sequences) if n is None else zip(*(np.random.choice(seq, n, replace=True) for seq in sequences)) 

def intensive_search(
    params: HiperparamGrid,
    preditor: SklearnModel,
    xtrain: DataFrame,
    ytrain: Series,
    xtest: DataFrame,
    ytest: Series,
    scores: Mapping[str, ScoreFunction],
    n_iters: Optional[int]=None,
    lesser_is_better: bool=True,
    **kwargs) -> tuple[SklearnModel, DataFrame]:
    
    """
    Performs an exhaustive search over a hyperparameter grid to find the best model.

    This function performs an intensive search over a defined hyperparameter grid to identify the best model 
    based on the evaluation scores. It generates combinations of hyperparameters, evaluates the model for each 
    combination, and tracks the best performing hyperparameters based on the given scoring functions.

    Parameters
    ----------
    params : HiperparamGrid
        A dictionary where the keys are hyperparameter names and the values are lists of possible values for each hyperparameter.
    preditor : SklearnModel
        The machine learning model (predictor) to be tuned. It should be an instance of a scikit-learn model.
    xtrain : DataFrame
        Training features used to fit the model.
    ytrain : Series
        Training labels corresponding to `xtrain`.
    xtest : DataFrame
        Test features used to evaluate the model.
    ytest : Series
        Test labels corresponding to `xtest`.
    n_iters : int, optional
        The number of hyperparameter combinations to evaluate. If not specified, the total number of combinations is used.
    scores : dict[str, ScoreFunction], optional
        A dictionary where keys are score names and values are scoring functions. The score functions are used to evaluate the model.
    lesser_is_better : bool, default=True
        Indicates if a lower score is better. If False, a higher score is better.
    verbose : bool, default=True
        If True, displays a progress bar showing the search progress.
    **kwargs
        Additional keyword arguments to be passed to the model or evaluation functions.

    Returns
    -------
    tuple[SklearnModel, DataFrame]
        A tuple containing:
        - The best model found during the search.
        - A DataFrame containing all evaluated hyperparameters and their corresponding scores, sorted by the main score.

    Notes
    -----
    - The function generates combinations of hyperparameters or samples from the parameter space.
    - The function creates and trains a model with given hyperparameters.
    - The function evaluates the model using the provided test data and scoring functions.
    - The function determine if a newly evaluated score is better than the current best score.
    - The provides a visual indication of the search progress.
    - The resulting DataFrame contains all hyperparameter trials and their corresponding evaluation scores, 
      with the rows sorted based on the main score.
    """

    parameter_trials: list[Mapping[str, Scalar]] = []

    total: int = n_iters or prod( len(it) for it in params.values() )
    search_space: Generator = searchSpace(*params.values(), n=n_iters)

    bestScore: float = None
    bestModel: SklearnModel = None

    mainScore_key: str = dictK(scores, 0)
    
    for sample in (pbar := progress_bar(search_space, total=total, desc='intensive Hiperparams Tune') ):

        params_i: Hiperparams = dictzip(params, sample)
        model = create_model(params_i, preditor, xtrain, ytrain)

        scores_i = evaluate_model(model, xtest, ytest, scores)
        candidate = scores_i.get(mainScore_key)

        if is_better_than(candidate, bestScore, lesser_is_better):
            bestModel, bestScore = model, candidate
            pbar.set_postfix_str(f'best_{mainScore_key}={bestScore: .6f}')

        parameter_trials.append({**params_i, **scores_i})

    search = DataFrame(parameter_trials).sort_values(by=mainScore_key, ascending=lesser_is_better, ignore_index=True)
    return bestModel, search
