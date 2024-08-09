The utilities within this directory were written in Python 2 by Professor Shin-Han Shiu

Steps to install Python 2 and create a virtual environment
```bash
# Install Python 2
wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18
tar xzf Python-2.7.18.tgz
cd Python-2.7.18/
vi README # installation instructions
./configure
make
sudo -i # login as root
conda deactivate # base env was activated
cd /home/seguraab/ara-kinase-prediction/code/utils_py2/Python-2.7.18/
make install
which python2 # check access
which python # also python 2.7.18
exit # logout from root

# Create the virtual environment
cd ..
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
sudo python2 get-pip.py
which pip2
pip2.7 install --user -U pip==20.3.4
pip2 install --user virtualenv==20.15.1
virtualenv --python=python utils_py2_env # create the virtual env
source utils_py2_env/bin/activate # activate the environment
```

Compile the python scripts within the utils_py2 folder
```bash
# with utils_py2_env activated:
echo $PYTHONPATH # should be empty
python -m py_compile *.py
mkdir bin/
mv *.pyc bin/
echo 'export PYTHONPATH=${PYTHONPATH}:/home/seguraab/ara-kinase-prediction/code/utils_py2/bin' >> /home/seguraab/.bashrc
echo 'alias utils-py2-env="source /home/seguraab/ara-kinase-prediction/code/utils_py2/utils_py2_env/bin/activate"' >> /home/seguraab/.bash_aliases
source /home/seguraab/.bashrc


```