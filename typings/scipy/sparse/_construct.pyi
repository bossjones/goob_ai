"""
This type stub file was generated by pyright.
"""

"""Functions to construct sparse matrices and arrays
"""
__docformat__ = ...
__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum', 'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag', 'diags_array', 'block_array', 'eye_array', 'random_array']
def spdiags(data, diags, m=..., n=..., format=...): # -> dia_matrix | Any:
    """
    Return a sparse matrix from diagonals.

    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : sequence of int or an int
        Diagonals to set:

        * k = 0  the main diagonal
        * k > 0  the kth upper diagonal
        * k < 0  the kth lower diagonal
    m, n : int, tuple, optional
        Shape of the result. If `n` is None and `m` is a given tuple,
        the shape is this tuple. If omitted, the matrix is square and
        its shape is len(data[0]).
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    See Also
    --------
    diags_array : more convenient form of this function
    diags : matrix version of diags_array
    dia_matrix : the sparse DIAgonal format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    ...

def diags_array(diagonals, /, *, offsets=..., shape=..., format=..., dtype=...): # -> dia_array | Any:
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square array large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse array format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the array.

    Notes
    -----
    The result from `diags_array` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 1.11

    Examples
    --------
    >>> from scipy.sparse import diags_array
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    ...

def diags(diagonals, offsets=..., shape=..., format=..., dtype=...): # -> dia_matrix | Any:
    """
    Construct a sparse matrix from diagonals.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    spdiags : construct matrix from diagonals
    diags_array : construct sparse array instead of sparse matrix

    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.

    The result from `diags` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 0.11

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    ...

def identity(n, dtype=..., format=...): # -> coo_array | coo_matrix | dia_array | Any | dia_matrix:
    """Identity matrix in sparse format

    Returns an identity matrix with shape (n,n) using a given
    sparse format and dtype. This differs from `eye_array` in
    that it has a square shape with ones only on the main diagonal.
    It is thus the multiplicative identity. `eye_array` allows
    rectangular shapes and the diagonal can be offset from the main one.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    n : int
        Shape of the identity matrix.
    dtype : dtype, optional
        Data type of the matrix
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import scipy as sp
    >>> sp.sparse.identity(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.identity(3, dtype='int8', format='dia')
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>
    >>> sp.sparse.eye_array(3, dtype='int8', format='dia')
    <3x3 sparse array of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    ...

def eye_array(m, n=..., *, k=..., dtype=..., format=...): # -> coo_array | coo_matrix | dia_array | Any | dia_matrix:
    """Identity matrix in sparse array format

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int or tuple of ints
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <3x3 sparse array of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    ...

def eye(m, n=..., k=..., dtype=..., format=...): # -> coo_array | coo_matrix | dia_array | Any | dia_matrix:
    """Sparse matrix with ones on diagonal

    Returns a sparse matrix (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the matrix.
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye(3, dtype=np.int8)
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
        with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    ...

def kron(A, B, format=...): # -> coo_array | Any | coo_matrix | bsr_array | bsr_matrix:
    """kronecker product of sparse matrices A and B

    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional (default: 'bsr' or 'coo')
        format of the result (e.g. "csr")
        If None, choose 'bsr' for relatively dense array and 'coo' for others

    Returns
    -------
    kronecker product in a sparse format.
    Returns a sparse matrix unless either A or B is a
    sparse array in which case returns a sparse array.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> A = sp.sparse.csr_array(np.array([[0, 2], [5, 0]]))
    >>> B = sp.sparse.csr_array(np.array([[1, 2], [3, 4]]))
    >>> sp.sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    >>> sp.sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    """
    ...

def kronsum(A, B, format=...): # -> coo_array | Any | coo_matrix | bsr_array | bsr_matrix:
    """kronecker sum of square sparse matrices A and B

    Kronecker sum of two sparse matrices is a sum of two Kronecker
    products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
    and B has shape (n,n) and I_m and I_n are identity matrices
    of shape (m,m) and (n,n), respectively.

    Parameters
    ----------
    A
        square matrix
    B
        square matrix
    format : str
        format of the result (e.g. "csr")

    Returns
    -------
    kronecker sum in a sparse matrix format

    """
    ...

def hstack(blocks, format=..., dtype=...): # -> csr_matrix | csc_matrix | csr_array | csc_array | coo_matrix | Any | coo_array:
    """
    Stack sparse matrices horizontally (column wise)

    Parameters
    ----------
    blocks
        sequence of sparse matrices with compatible shapes
    format : str
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block(hstack(blocks))` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    vstack : stack sparse matrices vertically (row wise)

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, hstack
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> hstack([A,B]).toarray()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    ...

def vstack(blocks, format=..., dtype=...): # -> csr_matrix | csc_matrix | csr_array | csc_array | coo_matrix | Any | coo_array:
    """
    Stack sparse arrays vertically (row wise)

    Parameters
    ----------
    blocks
        sequence of sparse arrays with compatible shapes
    format : str, optional
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block(vstack(blocks))` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    hstack : stack sparse matrices horizontally (column wise)

    Examples
    --------
    >>> from scipy.sparse import coo_array, vstack
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5, 6]])
    >>> vstack([A, B]).toarray()
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    ...

def bmat(blocks, format=..., dtype=...): # -> csr_matrix | csc_matrix | csr_array | csc_array | coo_matrix | Any | coo_array:
    """
    Build a sparse array or matrix from sparse sub-blocks

    Note: `block_array` is preferred over `bmat`. They are the same function
    except that `bmat` can return a deprecated sparse matrix.
    `bmat` returns a coo_matrix if none of the inputs are a sparse array.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``block_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    blocks : array_like
        Grid of sparse matrices with compatible shapes.
        An entry of None implies an all-zero matrix.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    bmat : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use `block_array()`.

    See Also
    --------
    block_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, bmat
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> bmat([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> bmat([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    ...

def block_array(blocks, *, format=..., dtype=...): # -> csr_matrix | csc_matrix | csr_array | csc_array | coo_matrix | Any | coo_array:
    """
    Build a sparse array from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse arrays with compatible shapes.
        An entry of None implies an all-zero array.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    block : sparse array

    See Also
    --------
    block_diag : specify blocks along the main diagonals
    diags : specify (possibly offset) diagonals

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_array
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_array([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> block_array([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    ...

def block_diag(mats, format=..., dtype=...): # -> coo_array | Any | coo_matrix:
    """
    Build a block diagonal sparse matrix or array from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices or arrays
        Input matrices or arrays.
    format : str, optional
        The sparse format of the result (e.g., "csr"). If not given, the result
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix or array
        If at least one input is a sparse array, the output is a sparse array.
        Otherwise the output is a sparse matrix.

    Notes
    -----

    .. versionadded:: 0.11.0

    See Also
    --------
    block_array
    diags_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_diag
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])

    """
    ...

def random_array(shape, *, density=..., format=..., dtype=..., random_state=..., data_sampler=...): # -> coo_array | Any:
    """Return a sparse array of uniformly random numbers in [0, 1)

    Returns a sparse array with the given shape and density
    where values are generated uniformly randomly in the range [0, 1).

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

    Parameters
    ----------
    shape : int or tuple of ints
        shape of the array
    density : real, optional (default: 0.01)
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional (default: 'coo')
        sparse matrix format.
    dtype : dtype, optional (default: np.float64)
        type of the returned matrix values.
    random_state : {None, int, `Generator`, `RandomState`}, optional
        A random number generator to determine nonzero structure. We recommend using
        a `numpy.random.Generator` manually provided for every call as it is much
        faster than RandomState.

        - If `None` (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If an int, a new ``Generator`` instance is used,
          seeded with the int.
        - If a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        This random state will be used for sampling `indices` (the sparsity
        structure), and by default for the data values too (see `data_sampler`).

    data_sampler : callable, optional (default depends on dtype)
        Sampler of random data values with keyword arg `size`.
        This function should take a single keyword argument `size` specifying
        the length of its returned ndarray. It is used to generate the nonzero
        values in the matrix after the locations of those values are chosen.
        By default, uniform [0, 1) random values are used unless `dtype` is
        an integer (default uniform integers from that dtype) or
        complex (default uniform over the unit square in the complex plane).
        For these, the `random_state` rng is used e.g. `rng.uniform(size=size)`.

    Returns
    -------
    res : sparse array

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import numpy as np
    >>> import scipy as sp
    >>> rng = np.random.default_rng()

    Default sampling uniformly from [0, 1):

    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, random_state=rng,
    ...                      data_sampler=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X().rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            random_state=rng, data_sampler=Y)
    """
    ...

def random(m, n, density=..., format=..., dtype=..., random_state=..., data_rvs=...): # -> coo_matrix | Any:
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    .. warning::

        Since numpy 1.17, passing a ``np.random.Generator`` (e.g.
        ``np.random.default_rng``) for ``random_state`` will lead to much
        faster execution times.

        A much slower implementation is used by default for backwards
        compatibility.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage of the
        sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        - If `seed` is None (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `seed` is an int, a new ``RandomState`` instance is used,
          seeded with `seed`.
        - If `seed` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        This random state will be used for sampling the sparsity structure, but
        not necessarily for sampling the values of the structurally nonzero
        entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Returns
    -------
    res : sparse matrix

    See Also
    --------
    random_array : constructs sparse arrays instead of sparse matrices

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import scipy as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, random_state=rng):
    ...     return random_state.standard_normal(size) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, random_state=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=random_state) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng,
    ...                      data_rvs=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return random_state.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = sp.sparse.random(3, 4, density=0.25, random_state=rng, data_rvs=Y.rvs)
    """
    ...

def rand(m, n, density=..., format=..., dtype=..., random_state=...): # -> coo_matrix | Any:
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    See Also
    --------
    random : Similar function allowing a custom random data sampler
    random_array : Similar to random() but returns a sparse array

    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
    >>> matrix
    <3x4 sparse matrix of type '<class 'numpy.float64'>'
       with 3 stored elements in Compressed Sparse Row format>
    >>> matrix.toarray()
    array([[0.05641158, 0.        , 0.        , 0.65088847],  # random
           [0.        , 0.        , 0.        , 0.14286682],
           [0.        , 0.        , 0.        , 0.        ]])

    """
    ...

