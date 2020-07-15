.. _timedomain:

Prepare data for time domain
============================

In order to mimic the realistic situation where only a limited number of observed epochs is available at each
day, it is necessary to prepare our simulate data to resemble this scenario. In ``resspect`` this is done in
5 steps:

1. Determine minimum and maximum MJD for the entire sample;

2. For each day of the survey, run through the entire data sample and select only the observed epochs which were obtained prior to it;

3. Perform the feature extraction process considering only the photometric points which survived item 2.

4. Check if, at each MJD in question, the object is available for querying.

5. Join all information in a standard features file.

Defining a queryable object and observation cost
------------------------------------------------

In the current implementation of ``resspect`` there are 2 ways in which you can determine if an object is available for query:  

1. Magnitude cut considering the last measured epoch, independently of how long ago it happened.  

2. Magnitude cut in the day of query, extrapolated using a chosen feature extraction method in case the last photometric point happened more than a number of days ago.


For both options, you can also choose to calculate the necessary telescope time to obtain the required spectra, given characteristics of a telescope and observation conditions. This is done using the `ExpTimeCalc class <https://resspect.readthedocs.io/en/latest/api/resspect.ExpTimeCalc.html>`_. This class was heavily based in the public `HiTS exposure time calculator <https://github.com/fforster/HiTS-public>`_.

In the most simple scenario, you can choose the diameter of the primary mirror for a given telescope, the magnitude of the object at the time of observation and the required SNR. All other default parameters are set to the DECam standards. Check the doc strings for more information on how to choose a different configuration.


For SNPCC
^^^^^^^^^

You can perform the entire analysis for one day of the survey using the `SNPCCPhotometry class <https://resspect.readthedocs.io/en/latest/api/resspect.SNPCCPhotometry.html>`_:

.. code-block:: python
   :linenos:

   >>> from resspect.time_domain_SNPCC import SNPCCPhotometry

   >>> path_to_data = 'data/SIMGEN_PUBLIC_DES/'
   >>> output_dir = 'results/time_domain/'
   >>> day = 20
   >>> queryable_criteria = 2
   >>> get_cost = True

   >>> data = SNPCCPhotometry()
   >>> data.create_daily_file(output_dir=output_dir, day=day)
   >>> data.build_one_epoch(raw_data_dir=path_to_data, day_of_survey=day,
   >>>                      time_domain_dir=output_dir, queryable_criteria=queryable_criteria, get_cost=get_cost)


Alternatively you can use the command line to prepare a sequence of days in one batch:

.. code-block:: bash

   >>> build_time_domain.py -d 20 21 22 23 -p <path to raw data dir> 
   >>>        -o <path to output time domain dir> -q 2 -c True

