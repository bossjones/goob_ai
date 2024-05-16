"""
This type stub file was generated by pyright.
"""

"""Utility functions to use Python Array API compatible libraries.

For the context about the Array API see:
https://data-apis.org/array-api/latest/purpose_and_scope.html

The SciPy use case of the Array API is described on the following page:
https://data-apis.org/array-api/latest/use_cases.html#use-case-scipy
"""
__all__ = ['array_namespace', '_asarray', 'size']
SCIPY_ARRAY_API: str | bool = ...
SCIPY_DEVICE = ...
_GLOBAL_CONFIG = ...
def compliance_scipy(arrays):
    """Raise exceptions on known-bad subclasses.

    The following subclasses are not supported and raise and error:
    - `numpy.ma.MaskedArray`
    - `numpy.matrix`
    - NumPy arrays which do not have a boolean or numerical dtype
    - Any array-like which is neither array API compatible nor coercible by NumPy
    - Any array-like which is coerced by NumPy to an unsupported dtype
    """
    ...

def array_namespace(*arrays): # -> Any:
    """Get the array API compatible namespace for the arrays xs.

    Parameters
    ----------
    *arrays : sequence of array_like
        Arrays used to infer the common namespace.

    Returns
    -------
    namespace : module
        Common namespace.

    Notes
    -----
    Thin wrapper around `array_api_compat.array_namespace`.

    1. Check for the global switch: SCIPY_ARRAY_API. This can also be accessed
       dynamically through ``_GLOBAL_CONFIG['SCIPY_ARRAY_API']``.
    2. `compliance_scipy` raise exceptions on known-bad subclasses. See
       its definition for more details.

    When the global switch is False, it defaults to the `numpy` namespace.
    In that case, there is no compliance check. This is a convenience to
    ease the adoption. Otherwise, arrays must comply with the new rules.
    """
    ...

def atleast_nd(x, *, ndim, xp=...):
    """Recursively expand the dimension to have at least `ndim`."""
    ...

def copy(x, *, xp=...):
    """
    Copies an array.

    Parameters
    ----------
    x : array

    xp : array_namespace

    Returns
    -------
    copy : array
        Copied array

    Notes
    -----
    This copy function does not offer all the semantics of `np.copy`, i.e. the
    `subok` and `order` keywords are not used.
    """
    ...

def is_numpy(xp): # -> bool:
    ...

def is_cupy(xp): # -> bool:
    ...

def is_torch(xp): # -> bool:
    ...

def xp_assert_equal(actual, desired, check_namespace=..., check_dtype=..., check_shape=..., err_msg=..., xp=...): # -> None:
    ...

def xp_assert_close(actual, desired, rtol=..., atol=..., check_namespace=..., check_dtype=..., check_shape=..., err_msg=..., xp=...): # -> None:
    ...

def xp_assert_less(actual, desired, check_namespace=..., check_dtype=..., check_shape=..., err_msg=..., verbose=..., xp=...): # -> None:
    ...

def cov(x, *, xp=...):
    ...

def xp_unsupported_param_msg(param): # -> str:
    ...

def is_complex(x, xp):
    ...

