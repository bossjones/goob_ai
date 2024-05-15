"""
This type stub file was generated by pyright.
"""

"""Abstract linear algebra library.

This module defines a class hierarchy that implements a kind of "lazy"
matrix representation, called the ``LinearOperator``. It can be used to do
linear algebra with extremely large sparse or structured matrices, without
representing those explicitly in memory. Such matrices can be added,
multiplied, transposed, etc.

As a motivating example, suppose you want have a matrix where almost all of
the elements have the value one. The standard sparse matrix representation
skips the storage of zeros, but not ones. By contrast, a LinearOperator is
able to represent such matrices efficiently. First, we need a compact way to
represent an all-ones matrix::

    >>> import numpy as np
    >>> from scipy.sparse.linalg._interface import LinearOperator
    >>> class Ones(LinearOperator):
    ...     def __init__(self, shape):
    ...         super().__init__(dtype=None, shape=shape)
    ...     def _matvec(self, x):
    ...         return np.repeat(x.sum(), self.shape[0])

Instances of this class emulate ``np.ones(shape)``, but using a constant
amount of storage, independent of ``shape``. The ``_matvec`` method specifies
how this linear operator multiplies with (operates on) a vector. We can now
add this operator to a sparse matrix that stores only offsets from one::

    >>> from scipy.sparse.linalg._interface import aslinearoperator
    >>> from scipy.sparse import csr_matrix
    >>> offsets = csr_matrix([[1, 0, 2], [0, -1, 0], [0, 0, 3]])
    >>> A = aslinearoperator(offsets) + Ones(offsets.shape)
    >>> A.dot([1, 2, 3])
    array([13,  4, 15])

The result is the same as that given by its dense, explicitly-stored
counterpart::

    >>> (np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])
    array([13,  4, 15])

Several algorithms in the ``scipy.sparse`` library are able to operate on
``LinearOperator`` instances.
"""
__all__ = ['LinearOperator', 'aslinearoperator']
class LinearOperator:
    """Common interface for performing matrix vector products

    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.

    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.

    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (M, K).

    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.

    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])

    """
    ndim = ...
    __array_ufunc__ = ...
    def __new__(cls, *args, **kwargs): # -> Self:
        ...
    
    def __init__(self, dtype, shape) -> None:
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        ...
    
    def matvec(self, x): # -> ndarray[Any, dtype[Any]] | ndarray[Any, Any]:
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """
        ...
    
    def rmatvec(self, x): # -> ndarray[Any, dtype[Any]] | ndarray[Any, Any]:
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """
        ...
    
    def matmat(self, X): # -> matrix[Any, dtype[Any]] | matrix[Any, Any] | ndarray[Any, dtype[Any]]:
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        ...
    
    def rmatmat(self, X): # -> matrix[Any, dtype[Any]] | matrix[Any, Any] | ndarray[Any, dtype[Any]] | Any:
        """Adjoint matrix-matrix multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array, or 2-d array.
        The default implementation defers to the adjoint.

        Parameters
        ----------
        X : {matrix, ndarray}
            A matrix or 2D array.

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or 2D array depending on the type of the input.

        Notes
        -----
        This rmatmat wraps the user-specified rmatmat routine.

        """
        ...
    
    def __call__(self, x):
        ...
    
    def __mul__(self, x): # -> _ProductLinearOperator | _ScaledLinearOperator | ndarray[Any, dtype[Any]] | ndarray[Any, Any] | matrix[Any, dtype[Any]] | matrix[Any, Any]:
        ...
    
    def __truediv__(self, other): # -> _ScaledLinearOperator:
        ...
    
    def dot(self, x): # -> _ProductLinearOperator | _ScaledLinearOperator | ndarray[Any, dtype[Any]] | ndarray[Any, Any] | matrix[Any, dtype[Any]] | matrix[Any, Any]:
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        ...
    
    def __matmul__(self, other): # -> _ProductLinearOperator | _ScaledLinearOperator | ndarray[Any, dtype[Any]] | ndarray[Any, Any] | matrix[Any, dtype[Any]] | matrix[Any, Any]:
        ...
    
    def __rmatmul__(self, other): # -> _ScaledLinearOperator | _ProductLinearOperator | Any:
        ...
    
    def __rmul__(self, x): # -> _ScaledLinearOperator | _ProductLinearOperator | Any:
        ...
    
    def __pow__(self, p): # -> _PowerLinearOperator | _NotImplementedType:
        ...
    
    def __add__(self, x): # -> _SumLinearOperator | _NotImplementedType:
        ...
    
    def __neg__(self): # -> _ScaledLinearOperator:
        ...
    
    def __sub__(self, x): # -> _SumLinearOperator | _NotImplementedType:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def adjoint(self): # -> _AdjointLinearOperator:
        """Hermitian adjoint.

        Returns the Hermitian adjoint of self, aka the Hermitian
        conjugate or Hermitian transpose. For a complex matrix, the
        Hermitian adjoint is equal to the conjugate transpose.

        Can be abbreviated self.H instead of self.adjoint().

        Returns
        -------
        A_H : LinearOperator
            Hermitian adjoint of self.
        """
        ...
    
    H = ...
    def transpose(self): # -> _TransposedLinearOperator:
        """Transpose this linear operator.

        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        ...
    
    T = ...


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""
    def __init__(self, shape, matvec, rmatvec=..., matmat=..., dtype=..., rmatmat=...) -> None:
        ...
    


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""
    def __init__(self, A) -> None:
        ...
    


class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""
    def __init__(self, A) -> None:
        ...
    


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B) -> None:
        ...
    


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B) -> None:
        ...
    


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha) -> None:
        ...
    


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p) -> None:
        ...
    


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A) -> None:
        ...
    


class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint) -> None:
        ...
    
    @property
    def dtype(self):
        ...
    


class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=...) -> None:
        ...
    


def aslinearoperator(A): # -> LinearOperator | MatrixLinearOperator:
    """Return A as a LinearOperator.

    'A' may be any of the following types:
     - ndarray
     - matrix
     - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
     - LinearOperator
     - An object with .shape and .matvec attributes

    See the LinearOperator documentation for additional information.

    Notes
    -----
    If 'A' has no .dtype attribute, the data type is determined by calling
    :func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
    call upon the linear operator creation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinearoperator(M)
    <2x3 MatrixLinearOperator with dtype=int32>
    """
    ...

