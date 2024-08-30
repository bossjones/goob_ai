## pyenv - :coffee: Getting Started

> https://raw.githubusercontent.com/Unstructured-IO/community/main/README.md
>
Goob_ai's open-source packages currently target Python 3.10. If you are using or contributing
to Goob_ai code, we encourage you to work with Python 3.10 in a virtual environment. You can
use the following instructions to get up and running with a Python 3.10 virtual environment
with `pyenv-virtualenv`:

#### Mac / Homebrew

1. Install `pyenv` with `brew install pyenv`.
2. Install `pyenv-virtualenv` with `brew install pyenv-virtualenv`
3. Follow the instructions [here](https://github.com/pyenv/pyenv#user-content-set-up-your-shell-environment-for-pyenv)
   to add the `pyenv-virtualenv` startup code to your terminal profile.
4. Install Python 3.10 by running `pyenv install 3.10.15`.
5. Create and activate a virtual environment by running:

```
pyenv virtualenv 3.10.15 unstructured
pyenv activate unstructured
```

You can changed the name of the virtual environment from `unstructured` to another name if you're
creating a virtual environment for a pipeline. For example, if you're a creating a virtual
environment for the SEC preprocessing, you can run `pyenv virtualenv 3.10.15 sec`.

#### Linux

1. Run `git clone https://github.com/pyenv/pyenv.git ~/.pyenv` to install `pyenv`
2. Run `git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv`
   to install `pyenv-virtualenv` as a `pyenv` plugin.
4. Follow steps 3-5 from the Mac/Homebrew instructions.
