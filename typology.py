import sys
sys.path.append('.')

from typing import Protocol, Union, NewType, Any
from numpy import ndarray

from collections.abc import Mapping, Iterable, Callable

# Define the score type as a callable that takes two ndarrays and returns a float
Array_ytrue = NewType('Array_ytrue', ndarray) # reference values
Array_ypred = NewType('Array_ypred', ndarray) # calculated values

ScoreFunction = Callable[[Array_ytrue, Array_ypred], float]  # Functions like f(y_true, y_calc) -> float

# Define LikeScore as either a dictionary of string to ScoreFunction, a sequence of ScoreFunctions, or a single ScoreFunction
ScoreLike = Union[Mapping[str, ScoreFunction], Iterable[ScoreFunction], ScoreFunction]

# Define Scalars and Arrays
Scalar = Union[str, float, int, complex]
ScalarOrIterable = Union[Scalar, Iterable]

# Define Hiperparameters
Hiperparams = Mapping[str, Scalar]
HiperparamGrid = Mapping[str, ndarray[Scalar]]

class SKlearnReescaler(Protocol):
    """
    Protocol for scikit-learn-like rescalers.

    Methods
    -------
    fit(X: Any, y: Any=None) -> 'SKlearnReescaler':
        Fit the rescaler to the data.
    
    transform(X: Any) -> ndarray:
        Apply the scaling transformation to the data.
    """
    def fit(self, X: Any, y: Any=None) -> 'SKlearnReescaler':
        ...

    def transform(self, X: Any) -> ndarray:
        ...

class SklearnModel(Protocol):
    """
    Protocol for scikit-learn-like regressors or classifiers.

    Methods
    -------
    fit(X: Any, y: Any) -> 'SklearnModel':
        Train the model using input features `X` and target `y`.
    
    predict(X: Any) -> ndarray:
        Predict target values for the input features `X`.
    """
    def fit(self, X: Any, y: Any) -> 'SklearnModel':
        ...

    def predict(self, X: Any) -> ndarray:
        ...

if __name__=='__main__':
    pass

