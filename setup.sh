# setup development environment (devenv)
python3 -m venv devenv
source devenv/bin/activate

# setup jupyter
pip install --upgrade pip
pip3 install jupyter

# install other requirements
pip install -r requirements.txt

# add devenv to jupyters ipykernel
pip3 install --user ipykernel
python -m ipykernel install --user --name=devenv

