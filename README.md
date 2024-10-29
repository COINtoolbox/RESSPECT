

<h1 align="center">
<img align="center" src="https://raw.githubusercontent.com/lsstdesc/resspect/main/docs/images/logo_small.png" width="500">
</h1><br>

# Recommendation System for Spectroscopic follow-up

[![resspect](https://img.shields.io/badge/COIN--Focus-RESSPECT-red)](http://cosmostatistics-initiative.org/resspect/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/RESSPECT/smoke-test.yml)](https://github.com/LSSTDESC/RESSPECT/actions/workflows/smoke-test.yml)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/LSSTDESC/RESSPECT/asv-main.yml?label=benchmarks)](https://LSSTDESC.github.io/RESSPECT/benchmarks)

This repository holds the pipeline of the RESSPECT project, built as part of the inter-collaboration activities developed by the Cosmostatistics Initiative ([COIN](cosmostatistics-initiative.org)) and the LSST Dark Energy Science Collaboration ([DESC](https://lsstdesc.org/)). It was forked from the original RESSPECT repo that can be found here.

This work grew from activities developed within the [COIN Residence Program #4](http://iaacoin.wix.com/crp2017), using as a starting point their [ActSNClass](https://github.com/COINtoolbox/ActSNClass) software. 

The active learning and telescope resources pipeline is described in [Kennamer et al, 2020](https://cosmostatistics-initiative.org/portfolio-item/resspect1/). The pre-processed data set used to obtain the results shown in the paper is available through zenodo at [de Souza et al., 2020](https://zenodo.org/record/4399109#.X-sL21lKhNg).

We kindly ask you to include the full citation for the above mentioned work if you use this material in your research.

Full documentation can be found at [readthedocs](https://lsst-resspect.readthedocs.io/en/latest/).

# Dependencies

### For main code:

 - python >= 3.10.10  
 - astropy >= 5.2.1  
 - matplotlib >= 3.7.0
 - numpy >= 1.24.2
 - pandas >= 1.5.3
 - progressbar2 >= 4.2.0
 - scikit_learn >= 1.2.1
 - scipy >= 1.10.0
 - seaborn >= 0.12.2
 - setuptools >= 65.5.0
 - xgboost >= 1.7.3
 - iminuit >= 1.20.0
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

    (RESSPECT) >>> pip install -e .

> You may choose to create your virtual environment within the folder of the repository. If you choose to do this, you must remember to exclude the virtual environment directory from version control using e.g., ``.gitignore``.  

# Starting the docker environment

The docker file in this repository can be run as a standalone environment for testing or developing RESSPECT, or in connection with tom. You will need to start by installing Docker Desktop for your chosen platform before you start. 

Note: These workflows have only been tested on macs; however the standalone docker image is built on Linux in in Github Actions CI.

## Using standalone

To use the container standalone first go into the root of the source directory, and build the container with:
```
docker build .
```

You can run the container two ways. The first way will use the version of resspect from your local checkout, which is probably what you want for development. After the
container is built run:
```
docker run -it --rm --mount type=bind,source=.,target=/resspect/resspect-src resspect
```

This will put you into a bash shell in the container with the venv for resspect already activated, and the current version of resspect in your source checkout installed.

If you wish to use the version of resspect packaged at build time, simply omit `--mount type=bind,source=.,target=/resspect/resspect-src` from the command above.

## Using with tom docker-compose setup

First checkout tom and follow [TOM's docker compose setup instructions](https://github.com/LSSTDESC/tom_desc?tab=readme-ov-file#deploying-a-dev-environment-with-docker). You will need to load [ELAsTiCC2 data](https://github.com/LSSTDESC/tom_desc?tab=readme-ov-file#for-elasticc2) into your tom environment in order to work with RESSPECT.

When you have finished that setup, go into the top level source directory and run these two commands:
```
docker compose build
docker compose up -d
```

You will now have a docker container called `resspect` which you can run resspect from. The version of resspect in use will be that of your local git checkout. That same docker container will be on the network with your tom setup, so you can access
the tom docker container on port 8080.

You can enter the `resspect` docker container to run commands with
```
docker compose run resspect
```

From the resspect container you should be able to log into the tom server with:
```
(resspect-venv) root@cd647ac7eca5:/resspect# python3
Python 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from resspect import tom_client as tc
>>> tc = tc.TomClient(url="http://tom:8080", username='admin', password='<your tom password>')
```
