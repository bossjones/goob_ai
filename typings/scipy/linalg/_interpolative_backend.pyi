"""
This type stub file was generated by pyright.
"""

"""
Direct wrappers for Fortran `id_dist` backend.
"""
_RETCODE_ERROR = ...
def id_srand(n):
    """
    Generate standard uniform pseudorandom numbers via a very efficient lagged
    Fibonacci method.

    :param n:
        Number of pseudorandom numbers to generate.
    :type n: int

    :return:
        Pseudorandom numbers.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def id_srandi(t): # -> None:
    """
    Initialize seed values for :func:`id_srand` (any appropriately random
    numbers will do).

    :param t:
        Array of 55 seed values.
    :type t: :class:`numpy.ndarray`
    """
    ...

def id_srando(): # -> None:
    """
    Reset seed values to their original values.
    """
    ...

def idd_frm(n, w, x):
    """
    Transform real vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idd_sfrm`, this routine works best when the length of
    the transformed vector is the power-of-two integer output by
    :func:`idd_frmi`, or when the length is not specified but instead
    determined a posteriori from the output. The returned transformed vector is
    randomly permuted.

    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idd_frmi`; `n` is also the length of the output vector.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_frmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_sfrm(l, n, w, x):
    """
    Transform real vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idd_frm`, this routine works best when the length of
    the transformed vector is known a priori.

    :param l:
        Length of transformed vector, satisfying `l <= n`.
    :type l: int
    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idd_sfrmi`.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_sfrmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_frmi(m):
    """
    Initialize data for :func:`idd_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_sfrmi(l, m):
    """
    Initialize data for :func:`idd_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddp_id(eps, A): # -> tuple[Any, Any, ndarray[Any, dtype[Any]]]:
    """
    Compute ID of a real matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_id(A, k): # -> tuple[Any, ndarray[Any, dtype[Any]]]:
    """
    Compute ID of a real matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_reconid(B, idx, proj): # -> ndarray[Any, dtype[Any]]:
    """
    Reconstruct matrix from real ID.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Reconstructed matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_reconint(idx, proj):
    """
    Reconstruct interpolation matrix from real ID.

    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Interpolation matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_copycols(A, k, idx):
    """
    Reconstruct skeleton matrix from real ID.

    :param A:
        Original matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`

    :return:
        Skeleton matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_id2svd(B, idx, proj): # -> tuple[Any, Any, Any]:
    """
    Convert real ID to SVD.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_snorm(m, n, matvect, matvec, its=...):
    """
    Estimate spectral norm of a real matrix by the randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate.
    :rtype: float
    """
    ...

def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=...):
    """
    Estimate spectral norm of the difference of two real matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the transpose of the first matrix to a vector, with
        call signature `y = matvect(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect: function
    :param matvect2:
        Function to apply the transpose of the second matrix to a vector, with
        call signature `y = matvect2(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect2: function
    :param matvec:
        Function to apply the first matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param matvec2:
        Function to apply the second matrix to a vector, with call signature
        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec2: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate of matrix difference.
    :rtype: float
    """
    ...

def iddr_svd(A, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddp_svd(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddp_aid(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute ID of a real matrix to a specified relative precision using random
    sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_estrank(eps, A):
    """
    Estimate rank of a real matrix to a specified relative precision using
    random sampling.

    The output rank is typically about 8 higher than the actual rank.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank estimate.
    :rtype: int
    """
    ...

def iddp_asvd(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified relative precision using random
    sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddp_rid(eps, m, n, matvect): # -> tuple[Any, Any, Any]:
    """
    Compute ID of a real matrix to a specified relative precision using random
    matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idd_findrank(eps, m, n, matvect):
    """
    Estimate rank of a real matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function

    :return:
        Rank estimate.
    :rtype: int
    """
    ...

def iddp_rsvd(eps, m, n, matvect, matvec): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified relative precision using random
    matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_aid(A, k): # -> tuple[Any, NDArray[Any] | Any]:
    """
    Compute ID of a real matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_aidi(m, n, k):
    """
    Initialize array for :func:`iddr_aid`.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Initialization array to be used by :func:`iddr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_asvd(A, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_rid(m, n, matvect, k): # -> tuple[Any, Any]:
    """
    Compute ID of a real matrix to a specified rank using random matrix-vector
    multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def iddr_rsvd(m, n, matvect, matvec, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a real matrix to a specified rank using random matrix-vector
    multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_frm(n, w, x):
    """
    Transform complex vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idz_sfrm`, this routine works best when the length of
    the transformed vector is the power-of-two integer output by
    :func:`idz_frmi`, or when the length is not specified but instead
    determined a posteriori from the output. The returned transformed vector is
    randomly permuted.

    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idz_frmi`; `n` is also the length of the output vector.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idz_frmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_sfrm(l, n, w, x):
    """
    Transform complex vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idz_frm`, this routine works best when the length of
    the transformed vector is known a priori.

    :param l:
        Length of transformed vector, satisfying `l <= n`.
    :type l: int
    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idz_sfrmi`.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_sfrmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_frmi(m):
    """
    Initialize data for :func:`idz_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_sfrmi(l, m):
    """
    Initialize data for :func:`idz_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzp_id(eps, A): # -> tuple[Any, Any, ndarray[Any, dtype[Any]]]:
    """
    Compute ID of a complex matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_id(A, k): # -> tuple[Any, ndarray[Any, dtype[Any]]]:
    """
    Compute ID of a complex matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_reconid(B, idx, proj): # -> ndarray[Any, dtype[Any]]:
    """
    Reconstruct matrix from complex ID.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Reconstructed matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_reconint(idx, proj):
    """
    Reconstruct interpolation matrix from complex ID.

    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Interpolation matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_copycols(A, k, idx):
    """
    Reconstruct skeleton matrix from complex ID.

    :param A:
        Original matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`

    :return:
        Skeleton matrix.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_id2svd(B, idx, proj): # -> tuple[Any, Any, Any]:
    """
    Convert complex ID to SVD.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_snorm(m, n, matveca, matvec, its=...):
    """
    Estimate spectral norm of a complex matrix by the randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate.
    :rtype: float
    """
    ...

def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=...):
    """
    Estimate spectral norm of the difference of two complex matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the adjoint of the first matrix to a vector, with
        call signature `y = matveca(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matveca: function
    :param matveca2:
        Function to apply the adjoint of the second matrix to a vector, with
        call signature `y = matveca2(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matveca2: function
    :param matvec:
        Function to apply the first matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param matvec2:
        Function to apply the second matrix to a vector, with call signature
        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec2: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate of matrix difference.
    :rtype: float
    """
    ...

def idzr_svd(A, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzp_svd(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzp_aid(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute ID of a complex matrix to a specified relative precision using
    random sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_estrank(eps, A):
    """
    Estimate rank of a complex matrix to a specified relative precision using
    random sampling.

    The output rank is typically about 8 higher than the actual rank.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank estimate.
    :rtype: int
    """
    ...

def idzp_asvd(eps, A): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified relative precision using
    random sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzp_rid(eps, m, n, matveca): # -> tuple[Any, Any, Any]:
    """
    Compute ID of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idz_findrank(eps, m, n, matveca):
    """
    Estimate rank of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function

    :return:
        Rank estimate.
    :rtype: int
    """
    ...

def idzp_rsvd(eps, m, n, matveca, matvec): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_aid(A, k): # -> tuple[Any, NDArray[Any] | Any]:
    """
    Compute ID of a complex matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_aidi(m, n, k):
    """
    Initialize array for :func:`idzr_aid`.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Initialization array to be used by :func:`idzr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_asvd(A, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_rid(m, n, matveca, k): # -> tuple[Any, Any]:
    """
    Compute ID of a complex matrix to a specified rank using random
    matrix-vector multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    ...

def idzr_rsvd(m, n, matveca, matvec, k): # -> tuple[Any, Any, Any]:
    """
    Compute SVD of a complex matrix to a specified rank using random
    matrix-vector multiplication.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    ...

