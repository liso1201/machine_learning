import sys
sys.path.append('.')

from typing import Optional
from typology import ScoreFunction, ScoreLike
from collections.abc import Mapping, Iterable

from model_utiles import SklearnModel

from sklearn.base import is_regressor, is_classifier
from sklearn.metrics import (
    r2_score, 
    max_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    mean_squared_error, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

#lesser_is_better score first!
regressor_metrics = (
    mean_squared_error,
    r2_score, 
    max_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error)
defaultRegressorScores: Mapping[str, ScoreFunction] = {f.__name__: f for f in regressor_metrics}

#lesser_is_better score first!
classifier_metrics = (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)
defaultClassifierScores: Mapping[str, ScoreFunction] = {f.__name__: f for f in classifier_metrics}

def is_iterable_of_callable(obj: object) -> bool:
    """
    Check if the given object is an iterable where all elements are callable.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if `obj` is an iterable where all elements are callable, otherwise False.

    Notes
    -----
    This function uses `isinstance` to check if the object is iterable and `all` with `map(callable, obj)`
    to ensure all elements are callable.
    """
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return all(map(callable, obj))
    return False

def map_scores(sc: ScoreLike) -> Mapping[str, ScoreFunction]:
    """
    Convert a score specification into a mapping of score names to score functions.

    Parameters
    ----------
    sc : ScoreLike
        A score specification which can be:
        - A dictionary where keys are score names and values are score functions.
        - An iterable of score functions.
        - A single score function.

    Returns
    -------
    Mapping[str, ScoreFunction]
        A mapping where the keys are the names of the score functions and the values are the score functions themselves.

    Notes
    -----
    - If `sc` is a dictionary, it is returned as is.
    - If `sc` is an iterable of score functions, each function is mapped to its name.
    - If `sc` is a single score function, it is mapped to its name.
    - The function uses `set` to ensure that duplicate functions are not included in the resulting mapping.
    """
    if isinstance(sc, dict):
        return sc.copy()
    elif is_iterable_of_callable(sc):
        return {f.__name__: f for f in set(sc)}
    elif callable(sc):
        return {sc.__name__: sc}
    else:
        raise ValueError('Invalid score specification. Expected a dictionary, an iterable of callable functions, or a single callable function.')


def provide_scores(scores: Optional[ScoreLike]=None, preditor: Optional[SklearnModel]=None) -> Mapping[str, ScoreFunction]:
    """
    Provide a mapping of score functions based on the provided scores or predictor.

    Parameters
    ----------
    scores : Optional[ScoreLike], default=None
        A score specification which can be:
        - A dictionary where keys are score names and values are score functions.
        - An iterable of score functions.
        - A single score function.
    preditor : Optional[SklearnModel], default=None
        A scikit-learn model to infer default scores from if `scores` is None.

    Returns
    -------
    Mapping[str, ScoreFunction]
        A mapping of score names to score functions.

    Raises
    ------
    ValueError
        If `scores` is None and `preditor` is not provided.
    TypeError
        If `preditor` is not a valid scikit-learn regressor or classifier when `scores` is None.

    Notes
    -----
    - If `scores` is None, the function checks the type of `preditor` to infer default scores:
        - If `preditor` is a regressor, default regressor scores are used.
        - If `preditor` is a classifier, default classifier scores are used.
    - The function convert the provided `scores` into a mapping.
    """
    if scores is None:
        if not preditor:
            raise ValueError('A valid SklearnModel must be provided if scores is None.')
        elif is_regressor(preditor):
            s = defaultRegressorScores
        elif is_classifier(preditor):
            s = defaultClassifierScores
        else:
            raise TypeError('The provided model must be a valid Sklearn regressor or classifier.')   
    else:
        s = map_scores(scores)

    return s
