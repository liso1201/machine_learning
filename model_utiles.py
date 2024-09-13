import sys
sys.path.append('.')

from pandas import DataFrame, Series
from typing import Any
from typology import Hiperparams, SklearnModel, ScoreFunction

def set_params(o: object, **params) -> object:
    """
    Set multiple attributes on an object based on provided parameters.

    Parameters
    ----------
    o : object
        The object on which attributes are to be set.
    **params : dict
        A dictionary of attribute names and values to be set on the object.

    Returns
    -------
    object
        The object with updated attributes.

    Notes
    -----
    Only attributes that already exist on the object will be set. Attributes
    that are not found in the object are ignored.
    """
    verify = lambda x: hasattr(o, x)
    for k in filter(verify, params):
        setattr(o, k, params.get(k))
    return o

def get_attributes(obj: object, *keys: str) -> dict[str, Any]:
    """
    Retrieve specified attributes from an object.

    Parameters
    ----------
    obj : object
        The object from which attributes are to be retrieved.
    *keys : str
        Names of the attributes to retrieve.

    Returns
    -------
    dict[str, Any]
        A dictionary of attribute names and their values. If an attribute is not 
        present on the object, its value will be `None`.

    Notes
    -----
    Only attributes that exist on the object are included in the result.
    """
    return {k: getattr(obj, k, None) for k in keys if hasattr(obj, k)}

def create_model(
        hiperparams: Hiperparams, 
        preditor: SklearnModel, 
        xtrain: DataFrame, 
        ytrain: Series) -> SklearnModel:
    """
    Create and train a machine learning model using the provided hyperparameters.

    Parameters
    ----------
    hiperparams : Hiperparams
        Dictionary of hyperparameters to be set on the model.
    preditor : SklearnModel
        The machine learning model to be trained.
    xtrain : DataFrame
        The training data features.
    ytrain : Series
        The training data target.

    Returns
    -------
    SklearnModel
        The trained machine learning model.

    Notes
    -----
    This function sets the hyperparameters on the model before fitting it to the data.
    """
    preditor = set_params(preditor, **hiperparams) 
    return preditor.fit(xtrain, ytrain)

def evaluate_model(
        preditor: SklearnModel, 
        xtest: DataFrame, 
        ytest: Series,
        scores: dict[str, ScoreFunction, float]) -> dict[str, float]:
    """
    Evaluate a machine learning model on a test dataset using multiple scoring functions.

    Parameters
    ----------
    preditor : SklearnModel
        The trained machine learning model to be evaluated.
    xtest : DataFrame
        The test data features.
    ytest : Series
        The test data target.
    scores : dict[str, Callable[[Array_ytrue, Array_ypred], float]]
        A dictionary of score functions. The keys are score names, and the values are 
        functions that take true and predicted values and return a score.

    Returns
    -------
    dict[str, float]
        A dictionary of score names and their computed values based on the test data.
    """
    ypred = preditor.predict(xtest)
    return {k: f(ytest, ypred) for k, f in scores.items() }



