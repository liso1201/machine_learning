import shelve

from dataclasses import dataclass
from sklearn import __version__ as skversion

from pathlib import Path

from typing import Any, Dict, Optional
from pandas import DataFrame

from typology import SklearnModel, SKlearnReescaler

@dataclass
class Artifact:
    """
    A class to represent an artifact, typically used for storing and retrieving
    machine learning model details and metadata.

    Attributes
    ----------
    data : Optional[DataFrame], default=None
        The dataset used in the model.
    variables : Optional[list[str]], default=None
        List of variable names in the dataset.
    target : Optional[str], default=None
        The target variable for the model.
    preditor : Optional[SklearnModel], default=None
        The trained machine learning model.
    hiperparams : Optional[Dict[str, Any]], default=None
        Hyperparameters used for model training.
    scaler : Optional[SKlearnReescaler], default=None
        Scaler or transformer used to preprocess the data.
    test_data : Optional[DataFrame], default=None
        Data used for testing the model.
    train_data : Optional[DataFrame], default=None
        Data used for training the model.
    n_iters : Optional[int], default=None
        Number of iterations performed during model training.
    hiperparams_search : Optional[DataFrame], default=None
        Results of hyperparameter search.
    sklearn_version : Optional[str], default=skversion
        The version of scikit-learn used when saving the artifact.
    """

    data: Optional[DataFrame] = None
    variables: Optional[list[str]] = None
    target: Optional[str] = None
    preditor: Optional[SklearnModel] = None
    hiperparams: Optional[Dict[str, Any]] = None
    scaler: Optional[SKlearnReescaler] = None
    test_data: Optional[DataFrame] = None
    train_data: Optional[DataFrame] = None
    n_iters: Optional[int] = None
    hiperparams_search: Optional[DataFrame] = None  
    sklearn_version: Optional[str] = skversion

    @classmethod
    def read_shelve(cls, dirname: str) -> 'Artifact':
        """
        Read an artifact from a shelve database.

        Parameters
        ----------
        dirname : str
            The name of the shelve database directory.

        Returns
        -------
        Artifact
            The artifact read from the shelve.

        Raises
        ------
        RuntimeError
            If there is an error reading from the shelve.
        ValueError
            If required attributes are missing from the shelve.
        """
        repository = Path(dirname)
        backup = 'backup'

        try:
            with shelve.open(repository / backup, 'r') as db:
                attributes = {attr: db.get(attr) for attr in cls.__annotations__}
        except Exception as e:
            raise RuntimeError(
                f'Failed to read from shelve: {e}.\n'
                f'This issue might be related to scikit-learn version incompatibility. '
                f'Consider using version {cls.sklearn_version}.\n'
                f'pip install scikit-learn=={cls.sklearn_version}.'
            )
        
        return Artifact(**attributes)

    def to_shelve(self, dirname: str) -> None:
        """
        Save the artifact to a shelve database.

        Parameters
        ----------
        dirname : str
            The name of the shelve database directory.

        Raises
        ------
        ValueError
            If the artifact does not have a name.
        """
        repository = Path(dirname)
        repository.mkdir(exist_ok=True, parents=True)
        
        backup = 'backup'
        with shelve.open(repository / backup, 'c') as new_db:
            for key, value in self.__dict__.items():
                new_db[key] = value

    @property
    def keys(self) -> list[str]:
        """
        List the attribute names (keys) of the artifact.

        Returns
        -------
        list of str
            The list of attribute names (keys).
        """
        return list(self.__annotations__)

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value of an attribute.

        Parameters
        ----------
        key : str
            The name of the attribute to retrieve.

        Returns
        -------
        Optional[Any]
            The value of the attribute, or None if it does not exist.
        """
        return self.__dict__.get(key, None)


if __name__ == '__main__':
    a = Artifact()
    print(a)