#!/bin/bash
set -e

exit_err() {
    echo "ERROR: $@" >&2
    exit 1
}

[[ "$(command -v git)" ]] || { exit_err "git is not installed."; }
[[ "$(command -v pip3)" ]] || { exit_err "pip3 is not installed."; }
[[ "$(command -v python3)" ]] || { exit_err "python3 is not installed."; }

PYTHON_PATH="$HOME/.pyenv/versions/3.8.0/bin/python3"

project_name="sdo-cli"
pip3 install -U virtualenv
virtualenv -p $PYTHON_PATH .venv
source .venv/bin/activate

python3 -m pip install -U pip
pip install -r requirements.txt
pip install -U jupyter_contrib_nbextensions

jupyter contrib nbextension install --sys-prefix
jupyter nbextension enable code_prettify/autopep8
ipython kernel install --user --name $project_name --display-name "Python ($project_name)"

python3 -m pip install --upgrade twine
