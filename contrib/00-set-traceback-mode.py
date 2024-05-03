# https://stackoverflow.com/questions/50557680/how-to-set-xmode-verbose-on-jupyter-notebook-at-launch
from IPython import get_ipython
ip = get_ipython()
ip.InteractiveTB.set_mode(mode="Verbose")
