import sys
sys.path.append('.')

import numpy as np

from collections.abc import Mapping, Callable
from typology import SklearnModel, Scalar, ScalarOrIterable, Hiperparams, HiperparamGrid, ScoreFunction
from typing import Optional, Union
from pandas import DataFrame, Series, concat

from utiles import kwargs_parser, dictK, dictV,is_better_than

from ui_tools import progress_bar
from model_utiles import create_model, evaluate_model

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import LabelEncoder

from functools import partial


# ==================
# CATEGORICAL TREATMENT
# ==================
def is_categorical(v: ScalarOrIterable) -> bool:
    """
    Determine if the input is a categorical variable (i.e., contains strings).

    Parameters
    ----------
    v : ScalarOrIterable
        A scalar or iterable value to check for categorical data.

    Returns
    -------
    bool
        True if the input is categorical (i.e., contains strings), otherwise False.
    
    Notes
    -----
    If the input is a scalar, the function checks if it's a string. If the input 
    is iterable, it checks if any element is a string.
    """
    if isinstance(v, Scalar):
        return isinstance(v, str)
    return any(isinstance(vi, str) for vi in v)

def force_intScalar_if_possible(v: Scalar) -> Scalar:
    """
    Convert a scalar value to an integer if possible, otherwise return the original value.

    Parameters
    ----------
    v : Scalar
        The scalar value to attempt to convert.

    Returns
    -------
    Scalar
        The converted integer value if possible, or the original value.
    
    Notes
    -----
    This function first tries to convert the input to an integer. If that fails, 
    it attempts to convert the input to a float and then to an integer. If both 
    conversions fail, the original value is returned.
    """
    try:
        return int(v)
    except:
        pass
    try:
        return int(float(v))
    except:
        return v

def force_intType(v: ScalarOrIterable) -> ScalarOrIterable:
    """
    Convert scalar or iterable values to integers if possible, otherwise return the original values.

    Parameters
    ----------
    v : ScalarOrIterable
        A scalar or iterable value to attempt to convert to integers.

    Returns
    -------
    ScalarOrIterable
        The converted integer value(s) if possible, or the original value(s).
    
    Notes
    -----
    For scalar inputs, this function attempts to convert the input to an integer.
    For iterable inputs, it applies the conversion to each element in the iterable.
    """
    if isinstance(v, Scalar):
        return force_intScalar_if_possible(v)
    else:
        return tuple(map(force_intScalar_if_possible, v))

def categorical_encode(
    h: Union[Hiperparams, HiperparamGrid], 
    encoders: Optional[Mapping[str, LabelEncoder]]=None, 
    inverse: bool=False,
    force_pandas: bool=False) -> Union[Hiperparams, HiperparamGrid, DataFrame]:
    """
    Encode or decode categorical hyperparameters using provided LabelEncoders.

    Parameters
    ----------
    h : Union[Hiperparams, HiperparamGrid]
        The hyperparameters to encode or decode.
    encoders : Optional[Mapping[str, LabelEncoder]], optional
        A dictionary of LabelEncoders for encoding or decoding each categorical feature, by default None.
    inverse : bool, optional
        If True, the function performs decoding (inverse transform). By default False.
    force_pandas : bool, optional
        If True, the function returns a pandas DataFrame, by default False.

    Returns
    -------
    Union[Hiperparams, HiperparamGrid, DataFrame]
        The encoded or decoded hyperparameters, potentially as a DataFrame if `force_pandas` is True.

    Notes
    -----
    If no encoders are provided, the input hyperparameters are returned unchanged. 
    If `force_pandas` is True, the result is converted to a pandas DataFrame.
    """
    if encoders is None:
        return h
    
    new: Union[Hiperparams, HiperparamGrid] = {}
    
    for k, v in h.items():

        if k in encoders:
            f: Callable = encoders[k].inverse_transform if inverse else encoders[k].transform
            v = force_intType(v)

            if isinstance(v, Scalar):
                new[k] = f([v])[0]
            else:
                new[k] = f(v)
        else:
            new[k] = v
    
    if force_pandas:
        return DataFrame(new)
    
    return new

def generate_hiperparams_LabelEncoders(h: HiperparamGrid) -> dict[str, LabelEncoder]:
    """
    Generate LabelEncoders for each categorical hyperparameter.

    Parameters
    ----------
    h : HiperparamGrid
        A dictionary of hyperparameter grids where some values may be categorical.

    Returns
    -------
    dict[str, LabelEncoder]
        A dictionary mapping hyperparameter names to their corresponding LabelEncoders.

    Notes
    -----
    This function only creates encoders for hyperparameters that are identified as categorical.
    """
    return {k: LabelEncoder().fit(v) for k, v in h.items() if is_categorical(v) }

# ==================
# AUXILIARY FUNCTIONS
# ==================
def score(
    x: Hiperparams,
    preditor: SklearnModel, 
    xtrain: DataFrame, 
    ytrain: Series, 
    xtest: DataFrame, 
    ytest: Series, 
    scores: dict[str, ScoreFunction]) -> Mapping[str, float]:
    """
    Train a model with the provided hyperparameters and compute evaluation metrics.

    Parameters
    ----------
    x : Hiperparams
        A dictionary of hyperparameters for the model.
    preditor : SklearnModel
        The predictive model to train.
    xtrain : DataFrame
        The training input data.
    ytrain : Series
        The training output data (target).
    xtest : DataFrame
        The test input data.
    ytest : Series
        The test output data (target).
    scores : dict[str, ScoreFunction]
        A dictionary where keys are metric names and values are functions that 
        compute the score.

    Returns
    -------
    Mapping[str, float]
        A dictionary mapping the metric name to the computed score.

    Notes
    -----
    The model is trained using the training data and then evaluated on the test data.
    """
    preditor = create_model(x, preditor, xtrain, ytrain)
    return evaluate_model(preditor, xtest, ytest, scores)

def continuom_samples(arr: np.ndarray, bins: Optional[int]=None, equiprobable: bool=True) -> np.ndarray:
    """
    Generate linearly spaced samples from discrete intervals in a continuous array.

    Parameters
    ----------
    arr : np.ndarray
        A 1D array of numerical values to sample from. The array will be sorted and used to define intervals.
    bins : int
        The number of samples to generate within each interval.
    equiprobable : bool, optional
        If True, return only unique samples to ensure that the samples are equiprobable. 
        Default is True.

    Returns
    -------
    np.ndarray
        A 1D array of linearly spaced samples between each pair of adjacent values in `arr`.
        Each interval between consecutive values in `arr` will contain `bins` evenly spaced points.
        If `equiprobable` is True, duplicates will be removed to ensure unique samples.

    Notes
    -----
    This function creates detailed, evenly spaced sampling between the discrete values in `arr`, 
    ensuring high resolution within each interval. If `equiprobable` is set to True, the function 
    will remove duplicate values to provide a unique set of samples across the intervals.

    Examples
    --------
    >>> continuom_samples(np.array([1, 4, 7]), bins=3)
    array([1. , 1.5, 2. , 4. , 4.5, 5. , 7. ])
    
    >>> continuom_samples(np.array([1, 4, 7]), bins=3, equiprobable=True)
    array([1. , 1.5, 2. , 4. , 5. , 7. ])
    """
    # Sort the array to ensure intervals are defined correctly
    sorted_arr = np.sort(arr)
    
    # Initialize an empty list to collect all samples
    all_samples = []

    if bins is None:
        bins = 10
    
    # Iterate over each pair of adjacent values in the sorted array
    for i in range(len(sorted_arr) - 1):
        # Append the samples to the list
        all_samples.append(
            np.linspace(sorted_arr[i], sorted_arr[i + 1], num=bins, endpoint=False, dtype=sorted_arr.dtype)
        ) 
    
    if equiprobable:
        return np.unique(np.concatenate(all_samples))

    return np.concatenate(all_samples)

def bayesian_hiperparamSpace(grid: HiperparamGrid, bins: Optional[int]=None, categoricals: Optional[list[str]]=None) -> HiperparamGrid:
    """
    Generate a hyperparameter space for Bayesian optimization.

    Parameters
    ----------
    grid : HiperparamGrid
        A dictionary representing the hyperparameter grid, where each key is a 
        hyperparameter and each value is a list of possible values.
    bins : Optional[int], optional
        The number of bins to use for continuous parameters, by default 100.
    categoricals : Optional[list[str]], optional
        A list of categorical parameters that should not be binned, by default None.

    Returns
    -------
    HiperparamGrid
        A new hyperparameter grid where continuous parameters are discretized into `bins` samples.
    
    Notes
    -----
    Continuous hyperparameters are sampled using `continuom_samples`, while categorical 
    parameters remain unchanged.
    """
    if categoricals is None:
        categoricals = []
    
    if bins is None:
        bins = 100
    
    new = {}
    
    for k, v in grid.items():
        new[k] = np.unique(v if k in categoricals else continuom_samples(v, bins))
        
    return new

def sample_generator(space: HiperparamGrid, n: Optional[int]=None) -> DataFrame:
    """
    Generate random samples from a hyperparameter space.

    Parameters
    ----------
    space : HiperparamGrid
        A dictionary representing the hyperparameter space.
    n : Optional[int], optional
        The number of samples to generate, by default 5.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing `n` random samples from the hyperparameter space.
    
    Notes
    -----
    Each hyperparameter is sampled independently from its set of possible values.
    """
    return DataFrame({k: np.random.choice(v, size=n or 5, replace=True) for k, v in space.items()})

def acquisition(
    gp: GaussianProcessRegressor, 
    space: HiperparamGrid, 
    kappa: Optional[float]=None,
    lesser_is_better: bool=True) -> Hiperparams:
    """
    Perform acquisition step in Bayesian optimization.

    Parameters
    ----------
    gp : GaussianProcessRegressor
        A trained Gaussian Process model used to predict the objective function.
    space : HiperparamGrid
        The hyperparameter search space.
    kappa : Optional[float], optional
        Exploration-exploitation trade-off parameter for Upper Confidence Bound (UCB), by default 2.0.
    lesser_is_better : bool, optional
        If True, the optimization problem is a minimization; otherwise, it's a maximization. 
        By default True.

    Returns
    -------
    Hiperparams
        The hyperparameter set corresponding to the best acquisition function value.
    
    Notes
    -----
    The acquisition function used is the Upper Confidence Bound (UCB). For minimization, 
    it selects the hyperparameters that minimize the predicted value; for maximization, 
    it selects those that maximize it.
    """
    x_samples: DataFrame = sample_generator(space, n=1000).astype('O')
    
    y_avg, y_std = gp.predict(x_samples.values, return_std=True)
    y_std: np.ndarray = y_std + 1e-6

    if kappa is None:
        kappa = 2.0

    if lesser_is_better:
        # Minimization: lower y_avg is better
        improvement: np.ndarray = y_avg.min() - y_avg
    else:
        # Maximization: higher y_avg is better
        improvement: np.ndarray = y_avg - y_avg.max()

    ucbx: int = np.argmax(improvement + kappa * y_std)
    return x_samples.loc[ucbx].to_dict()

# ==================
# MAIN FUNCTION
# ==================
def bayesian_search(
    grid: HiperparamGrid,
    preditor: SklearnModel,
    xtrain: DataFrame,
    ytrain: Series,
    xtest: DataFrame,
    ytest: Series,
    scores: Mapping[str, ScoreFunction],
    n_iters: Optional[int]=None,
    lesser_is_better: bool=True,
    n_samples: Optional[int]=None,
    kernel:Optional[Kernel]=None,
    kappa:Optional[float]=None,
    bins: Optional[int]=None,
    **kwargs) -> tuple[SklearnModel, DataFrame]:
    '''
    Perform Bayesian optimization for hyperparameter tuning of a machine learning model.

    This function performs a Bayesian optimization search over the specified hyperparameter grid to 
    find the best set of hyperparameters for a given model. The search process involves defining a 
    surrogate model (Gaussian Process) and iteratively improving the hyperparameters based on the 
    acquisition function.

    Parameters
    ----------
    grid : HiperparamGrid
        A dictionary mapping hyperparameter names to their possible values, or a numpy array of possible values.
    preditor : SklearnModel
        An instance of a scikit-learn-like model to be optimized.
    xtrain : DataFrame
        The training feature data.
    ytrain : Series
        The training target data.
    xtest : DataFrame
        The testing feature data.
    ytest : Series
        The testing target data.
    n_iters : Optional[int], default=None
        The number of iterations for the Bayesian optimization process. If None, it will be set to a default value.
    scores : Optional[Mapping[str, ScoreFunction]], default=None
        A dictionary of score functions to evaluate the model's performance. Each score function should
        take true and predicted values and return a float score.
    lesser_is_better : bool, default=True
        If True, lower scores are considered better. If False, higher scores are considered better.
    verbose : bool, default=True
        If True, progress will be shown during the optimization process.
    n_samples : Optional[int], default=None
        The number of samples to generate in the hyperparameter space for initial Gaussian Process fitting.
    kernel : Optional[Kernel], default=None
        The kernel to use for the Gaussian Process. If None, a default kernel is used.
    kappa : Optional[float], default=None
        The kappa parameter for the acquisition function. It balances exploration and exploitation.
    bins : Optional[int], default=None
        The number of bins to use when creating the hyperparameter space. If None, a default value is used.
    **kwargs : Additional keyword arguments
        Additional parameters to pass to the `GaussianProcessRegressor`.

    Returns
    -------
    tuple
        A tuple containing:
        - SklearnModel: The best model found during the optimization.
        - DataFrame: A DataFrame containing the results of the hyperparameter search.

    Notes
    -----
    The Bayesian optimization process involves:
    1. **Processing Categorical Variables**: Categorical variables in the hyperparameter grid are encoded using `LabelEncoder`.
    
    2. **Defining the Objective Function**: An objective function is defined to evaluate the model's performance based on hyperparameters. 
    The metric to be minimized (or maximized) is always the first callable in the `scores` dictionary; this is considered the most important metric.
    
    3. **Generating the Search Space**: The hyperparameter space is created, considering categorical variables and possible value ranges.
    
    4. **Sampling and Initial Fitting**: Initial samples are generated and used to fit the Gaussian Process model.
    
    5. **Optimization Loop**: The model iteratively improves the hyperparameters based on the acquisition function, which balances exploration and exploitation.
    
    6. **Results Extraction**: After the optimization loop, the best hyperparameters and corresponding model are extracted and returned.

    The search space considers only the minimum and maximum values of each hyperparameter to define the continuous ranges for Bayesian optimization. This ensures that the optimization process effectively explores the hyperparameter space and finds the best possible configuration.
    '''

    # processing categoricals
    encoders = generate_hiperparams_LabelEncoders(grid)
    
    # define objective function
    partial_input = dict(preditor=preditor, xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest, scores=scores)
    objective_function = partial(score, **partial_input)

    # generate search space
    space: HiperparamGrid = bayesian_hiperparamSpace(grid, bins=bins, categoricals=encoders)
    
    # calculated small size samples (x, y)
    x: DataFrame = sample_generator(space, n=n_samples)
    y: DataFrame = x.astype('O').apply(objective_function, axis=1, result_type='expand')

    # define kernel
    if kernel is None:
        kernel = C(1.0) * RBF(length_scale=1.0)
    
    # define GP model
    kw = kwargs_parser(GaussianProcessRegressor, kwargs, ignore=locals() )
    gp = GaussianProcessRegressor(kernel=kernel, **kw)

    # remove categorical strings: categoricals (str) -> numeric (int)
    space: HiperparamGrid = categorical_encode(space, encoders)
    x = categorical_encode(x, encoders, force_pandas=True)

    # setup score variables
    mainScore_key: str = dictK(scores, 0)
    bestScore: float = None

    #starting loop
    for _ in (pbar := progress_bar(total=n_iters, desc='Bayesian Hiperparams Tune') ):

        surrogate = gp.fit(x.values, y.iloc[:,0].values)

        xi: Hiperparams = acquisition(gp=surrogate, space=space, lesser_is_better=lesser_is_better, kappa=kappa)
        yi: dict[str, ScoreFunction] = objective_function(
            categorical_encode(xi, encoders, inverse=True) #decode xi
        )
        candidate = dictV(yi, 0)

        if is_better_than(candidate, bestScore, lesser_is_better):
            bestScore = candidate
            pbar.set_postfix_str(f'best_{mainScore_key}={bestScore: .6f}')

        x.loc[x.index.size] = xi
        y.loc[y.index.size] = yi

    # hiperparameters search
    search = concat((x,y), axis=1).astype(x.dtypes)
    search = search.sort_values(by=mainScore_key, ascending=lesser_is_better, ignore_index=True)
    
    # decode categorical variables in search and get best parameters from it
    search = categorical_encode(search, encoders, inverse=True, force_pandas=True)
    best_x: Hiperparams = search.iloc[0].to_dict()
    
    #bestModel, search
    return create_model(best_x, preditor, xtrain, ytrain), search


if __name__ == '__main__':
    pass