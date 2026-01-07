"""
Utility decorators for the newt package.

Provides common decorators for validation and error handling.
"""

from functools import wraps
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def requires_fit(attr_name: str = "is_fitted_") -> Callable[[F], F]:
    """
    Decorator that ensures the instance is fitted before method execution.

    Parameters
    ----------
    attr_name : str
        Name of the attribute that indicates fit status. Default "is_fitted_".

    Returns
    -------
    Callable
        Decorated method that raises ValueError if not fitted.

    Examples
    --------
    >>> class MyTransformer:
    ...     def __init__(self):
    ...         self.is_fitted_ = False
    ...
    ...     def fit(self, X):
    ...         self.is_fitted_ = True
    ...         return self
    ...
    ...     @requires_fit()
    ...     def transform(self, X):
    ...         return X * 2
    """

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, attr_name, False):
                class_name = self.__class__.__name__
                raise ValueError(f"{class_name} is not fitted. Call fit() first.")
            return method(self, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
