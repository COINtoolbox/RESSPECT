[![resspect](https://img.shields.io/badge/COIN--Focus-RESSPECT-red)](http://cosmostatistics-initiative.org/resspect/)

# <img align="right" src="docs/images/logo_small.png" width="350"> RESSPECT


## Recommendation System for Spectroscopic follow-up 

This repository holds the pipeline of the RESSPECT project, built as part of the inter-collaboration activities developed by the Cosmostatistics Initiative ([COIN](cosmostatistics-initiative.org)) and the LSST Dark Energy Science Collaboration ([DESC](https://lsstdesc.org/)).

This work grew from activities developed within the [COIN Residence Program #4](http://iaacoin.wix.com/crp2017), using as a starting point their [ActSNClass](https://github.com/COINtoolbox/ActSNClass) software. 

The active learning and telescope resources pipeline is described in [Kennamer et al, 2020](https://cosmostatistics-initiative.org/portfolio-item/resspect1/). The pre-processed data set used to obtain the results shown in the paper is available through zenodo at [de Souza et al., 2020](https://zenodo.org/record/4399109#.X-sL21lKhNg).

We kindly ask you to include the full citation for the above mentioned work if you use this material in your research.

Full documentation can be found at [readthedocs](https://resspect.readthedocs.io/en/latest/).

# Dependencies

### For main code:

 - python>=3.7  
 - astropy>=4.2.1  
 - matplotlib>=3.4.1
 - numpy>=1.19.5
 - pandas>=1.2.4
 - progressbar2>=3.53.1
 - pytest>=6.2.3
 - scikit_learn>=1.0.2
 - scipy>=1.6.2
 - seaborn>=0.11.1
 - setuptools>=49.2.1
 - xgboost>=1.4.0
 
### For cosmology metric:

 - cmdstanpy>=1.1.0
 
### For documentation:
 
  - sphinx>=2.1.2

# Install

The current version runs in Python-3.7 or higher and it was not tested on Windows.  

We recommend that you work within a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).  
 
You will need to install the `Python` package ``virtualenv``. In MacOS or Linux, do

    >> python3 -m pip install --user virtualenv

Navigate to a ``env_directory`` where you will store the new virtual environment and create it  

    >> python3 -m venv RESSPECT

> Make sure you deactivate any ``conda`` environment you might have running before moving forward.   

Once the environment is set up you can activate it,

    >> source <env_directory>/bin/activate

You should see a ``(RESSPECT)`` flag in the extreme left of terminal command line.   

Next, clone this repository in another chosen location:  

    (RESSPECT) >> git clone https://github.com/COINtoolbox/RESSPECT.git

Navigate to the repository folder and do  

    (RESSPECT) >> pip install -r requirements.txt


You can now install this package with:  

    (RESSPECT) >>> python setup.py install

> You may choose to create your virtual environment within the folder of the repository. If you choose to do this, you must remember to exclude the virtual environment directory from version control using e.g., ``.gitignore``.   
