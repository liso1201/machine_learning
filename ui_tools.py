import logging

from tqdm import tqdm
from typing import Optional, Iterable

def progress_bar(iterator: Optional[Iterable]=None, total: Optional[int]=None, **tqdm_karg) -> Optional[tqdm]:
    
    template = '{desc}: |{bar}| {n_fmt}/{total_fmt}{postfix}'

    if iterator is None:
        if total:
            iterator = range(total)
        else:
            raise ValueError('if iterator is None then total should be passed!')

    return tqdm(iterator, total=total, ascii=' =', colour='MAGENTA', bar_format=template, leave=True, **tqdm_karg)

def verbalise(txt: str, verbose: bool=True) -> None:
    if verbose:
        print(txt)
    return