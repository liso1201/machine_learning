import inspect
from typing import Callable, Optional, Iterable, Literal, Union, Any


def dictzip(k: Iterable, v: Iterable) -> dict[Any, Any]:
    """
    Create a dictionary from two iterable collections.

    Parameters
    ----------
    k : Iterable
        Iterable collection of keys.
    v : Iterable
        Iterable collection of values.

    Returns
    -------
    dict[Any, Any]
        A dictionary where the keys come from `k` and the values come from `v`.

    Examples
    --------
    >>> dictzip(['a', 'b', 'c'], [1, 2, 3])
    {'a': 1, 'b': 2, 'c': 3}
    """
    return dict(zip(k, v))

def dictK(d: dict, i: Optional[Union[int, slice, Literal['-']]]=None) -> Any:
    """
    Get a specific key or all keys from a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary from which keys will be extracted.
    i : Optional[Union[int, slice, Literal['-']]]
        Index or slice of keys to return. If '-', returns all keys.

    Returns
    -------
    Any
        The requested key or a list of all keys from the dictionary.

    Examples
    --------
    >>> dictK({'a': 1, 'b': 2, 'c': 3}, 1)
    'b'
    >>> dictK({'a': 1, 'b': 2, 'c': 3}, '-')
    ['a', 'b', 'c']
    """
    ll = list(d.keys())
    if i == '-':
        return ll
    else:
        return ll[i or 0]

def dictV(d: dict, i: Optional[Union[int, slice, Literal['-']]]=None) -> Any:
    """
    Get a specific value or all values from a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary from which values will be extracted.
    i : Optional[Union[int, slice, Literal['-']]]
        Index or slice of values to return. If '-', returns all values.

    Returns
    -------
    Any
        The requested value or a list of all values from the dictionary.

    Examples
    --------
    >>> dictV({'a': 1, 'b': 2, 'c': 3}, 1)
    2
    >>> dictV({'a': 1, 'b': 2, 'c': 3}, '-')
    [1, 2, 3]
    """
    ll = list(d.values())
    if i == '-':
        return ll
    else:
        return ll[i or 0]

def is_better_than(candidate: float, best: float=None, lesser_is_better: bool = True) -> bool:
    """
    Determine if the candidate is better than the best known value based on the comparison metric.

    Parameters
    ----------
    candidate : float
        The value to be compared.
    best : Optional[float]
        The best known value. If None, the candidate is considered better.
    lesser_is_better : bool, default=True
        If True, a smaller value is considered better. Otherwise, a larger value is considered better.

    Returns
    -------
    bool
        Returns True if the candidate is better according to the specified metric.

    Examples
    --------
    >>> is_better_than(3.0, 4.0)
    True
    >>> is_better_than(5.0, 4.0, lesser_is_better=False)
    False
    """
    if best is None:
        return True
    return candidate < best if lesser_is_better else candidate > best

def kwargs_parser(
        function: Callable, 
        kwargs: dict[str, Any], 
        defaults: Optional[dict[str, Any]]=None,
        ignore: Optional[list]=None,
        restricted: bool=True) -> dict[str, Any]:
    """
    Filters and adapts provided arguments to match the signature of a function.

    Parameters
    ----------
    function : Callable
        The function whose signature will be used to validate the arguments.
    kwargs : dict[str, Any]
        Dictionary of provided arguments.
    defaults : Optional[dict[str, Any]], default=None
        Dictionary of default values for arguments. If None, uses function's default values.
    ignore : Optional[list], default=None
        List of arguments to ignore. If None, no arguments are ignored.
    restricted : bool, default=True
        If True, only allows arguments that are in the function's signature. Otherwise, allows additional arguments.

    Returns
    -------
    dict[str, Any]
        A dictionary of filtered and adapted arguments to match the function's signature.

    Examples
    --------
    >>> def foo(a, b=2, c=3):
    ...     pass
    >>> kwargs_parser(foo, {'a': 1, 'b': 4, 'd': 5})
    {'a': 1, 'b': 4}
    """
    if defaults is None:
        defaults = {}
    
    if not isinstance(ignore, Iterable):
        ignore = list()
    
    kw: dict[str, Any] = {}
    for k, v in inspect.signature(function).parameters.items():
        if k in ignore:
            continue
        else:
            kw[k] = kwargs.pop(k, defaults.get(k, v.default) )
    
    if not restricted:
        kw |= kwargs

    return kw
