import better_exceptions, sys, types
from IPython import get_ipython
ip = get_ipython()
old_show = ip.showtraceback
def exception_thunk(self, exc_tuple=None, filename=None,
                    tb_offset=None, exception_only=False, **kwargs):

    notuple = False
    if exc_tuple is None:
        notuple = True
        exc_tuple = sys.exc_info()
    etype, value, tb = self._get_exc_info(exc_tuple)
    use_better = not any ([filename, tb_offset, exception_only, issubclass(etype, SyntaxError)])
    if use_better:
        return better_exceptions.excepthook(etype, value, tb)
    else:
        return old_show(None if notuple else exc_tuple,
                        filename, tb_offset, exception_only, **kwargs)

ip.showtraceback = types.MethodType(exception_thunk, ip)
