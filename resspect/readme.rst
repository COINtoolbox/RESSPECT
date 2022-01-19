resspect
========

*Recommendation System for Spectroscopic Follow-up*


List of modules:

:batch_functions.py: functions for batch strategies
:bazin.py: functions to find best-fit parameters to the Bazin function
:build_plasticc_canonical.py: constructs the canonical sample for PLAsTiCC data
:build_snpcc_canonical.py: constructs the canonical sample for SNPCC data
:classifiers.py: machine learning classifiers
:cosmo_metrics_utils.py: cosmology metric and auxiliary functions
:database.py: DataBase object upon which the learning is updated
:exposure_time_calculator.py: exposure time calculator
:fit_lightcurves.py: LightCurve object, perform fit on all samples
:learn_loop.py: active learning loop for the full light curve analysis
:metrics.py: metrics to evaluate classification results
:plot_results.py: plot diagnostics
:query_budget_strategies.py: strategies which optimize batch queries
:query_strategies.py: strategies to choose which object to query
:salt3_utils.py: auxiliary files for SNANA fits
:snana_fits_to_pd: converts SNANA FITS files to pandas.DataFrame
:time_domain_loop.py: Active Learning loop for time domain analysis
:time_domain_plasticc.py: Prepare PLAsTiCC data for time domain analysis
:time_domain_snpcc.py: Prepare data for time domain analysis


List of folders:


:example_scripts: Holds simple examples of different functionalities
:snana_pipe: Holds auxiliary files for SNANA
:scripts: Holds executables
:tests: Holds pipeline tests

