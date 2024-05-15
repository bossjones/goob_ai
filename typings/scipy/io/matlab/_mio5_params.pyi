"""
This type stub file was generated by pyright.
"""

import numpy as np

''' Constants and classes for matlab 5 read and write

See also mio5_utils.pyx where these same constants arise as c enums.

If you make changes in this file, don't forget to change mio5_utils.pyx
'''
__all__ = ['MDTYPES', 'MatlabFunction', 'MatlabObject', 'MatlabOpaque', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'OPAQUE_DTYPE', 'codecs_template', 'mat_struct', 'mclass_dtypes_template', 'mclass_info', 'mdtypes_template', 'miCOMPRESSED', 'miDOUBLE', 'miINT16', 'miINT32', 'miINT64', 'miINT8', 'miMATRIX', 'miSINGLE', 'miUINT16', 'miUINT32', 'miUINT64', 'miUINT8', 'miUTF16', 'miUTF32', 'miUTF8', 'mxCELL_CLASS', 'mxCHAR_CLASS', 'mxDOUBLE_CLASS', 'mxFUNCTION_CLASS', 'mxINT16_CLASS', 'mxINT32_CLASS', 'mxINT64_CLASS', 'mxINT8_CLASS', 'mxOBJECT_CLASS', 'mxOBJECT_CLASS_FROM_MATRIX_H', 'mxOPAQUE_CLASS', 'mxSINGLE_CLASS', 'mxSPARSE_CLASS', 'mxSTRUCT_CLASS', 'mxUINT16_CLASS', 'mxUINT32_CLASS', 'mxUINT64_CLASS', 'mxUINT8_CLASS']
miINT8 = ...
miUINT8 = ...
miINT16 = ...
miUINT16 = ...
miINT32 = ...
miUINT32 = ...
miSINGLE = ...
miDOUBLE = ...
miINT64 = ...
miUINT64 = ...
miMATRIX = ...
miCOMPRESSED = ...
miUTF8 = ...
miUTF16 = ...
miUTF32 = ...
mxCELL_CLASS = ...
mxSTRUCT_CLASS = ...
mxOBJECT_CLASS = ...
mxCHAR_CLASS = ...
mxSPARSE_CLASS = ...
mxDOUBLE_CLASS = ...
mxSINGLE_CLASS = ...
mxINT8_CLASS = ...
mxUINT8_CLASS = ...
mxINT16_CLASS = ...
mxUINT16_CLASS = ...
mxINT32_CLASS = ...
mxUINT32_CLASS = ...
mxINT64_CLASS = ...
mxUINT64_CLASS = ...
mxFUNCTION_CLASS = ...
mxOPAQUE_CLASS = ...
mxOBJECT_CLASS_FROM_MATRIX_H = ...
mdtypes_template = ...
mclass_dtypes_template = ...
mclass_info = ...
NP_TO_MTYPES = ...
NP_TO_MXTYPES = ...
codecs_template = ...
MDTYPES = ...
class mat_struct:
    """Placeholder for holding read data from structs.

    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.loadmat` function.
    """
    ...


class MatlabObject(np.ndarray):
    """Subclass of ndarray to signal this is a matlab object.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be instantiated directly.
    """
    def __new__(cls, input_array, classname=...): # -> Self:
        ...
    
    def __array_finalize__(self, obj): # -> None:
        ...
    


class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """
    def __new__(cls, input_array): # -> Self:
        ...
    


class MatlabOpaque(np.ndarray):
    """Subclass for a MATLAB opaque matrix.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be directly instantiated.
    """
    def __new__(cls, input_array): # -> Self:
        ...
    


OPAQUE_DTYPE = ...
