resspect
========

*Recommendation System for Spectroscopic Follow-up*


List of modules:

:bazin.py: functions to find best-fit parameters to the Bazin function
:build_snpcc_canonical: constructs the canonical sample
:classifiers.py: machine learning classifiers
:database.py: DataBase object upon which the learning is updated
:exposure_time_calculator.py: exposure time calculator
:fit_lightcurves.py: LightCurve object, perform fit on all samples
:learn_loop.py: active learning loop for the full light curve analysis
:metrics.py: metrics to evaluate classification results
:plot_results.py: plot diagnostics
:query_strategies.py: strategies to choose which object to query
:snana_fits_to_pd: converts SNANA FITS files to pandas.DataFrame
:time_domain_SNPCC.py: Prepare data for time domain analysis
:time_domain_loop.py: Active Learning loop for time domain analysis
:time_domain_PLAsTiCC.py: Prepare PLAsTiCC data for time domain analysis


List of folders:


:scripts: Holds executables
