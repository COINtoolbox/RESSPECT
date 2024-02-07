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

In the most simple scenario, you can choose the diameter of the primary mirror for a given telescope, the magnitude of the object at the time of observation and the required SNR. All other default parameters are set to the DECam standards. 


For SNPCC
^^^^^^^^^

You can perform the entire analysis for one day of the survey using the `SNPCCPhotometry class <https://resspect.readthedocs.io/en/latest/api/resspect.SNPCCPhotometry.html>`_:

.. code-block:: python
   :linenos:

   >>> from resspect.time_domain_snpcc import SNPCCPhotometry

   >>> path_to_data = 'data/SIMGEN_PUBLIC_DES/'
   >>> output_dir = 'results/time_domain/'
   >>> day = 20
   >>> queryable_criteria = 2
   >>> get_cost = True
   >>> feature_extractor = 'bazin'
   >>> tel_sizes=[4, 8]
   >>> tel_names = ['4m', '8m']
   >>> spec_SNR = 10
   >>> number_of_processors = 5

   >>> data = SNPCCPhotometry()
   >>> data.create_daily_file(output_dir=output_dir, day=day,
   >>>                        get_cost=get_cost)
   >>> data.build_one_epoch(raw_data_dir=path_to_data, 
   >>>                      day_of_survey=day, time_domain_dir=output_dir, 
   >>>                      feature_extractor=feature_extractor,
   >>>                      queryable_criteria=queryable_criteria, 
   >>>                      get_cost=get_cost, tel_sizes=tel_sizes, 
   >>>                      tel_names=tel_names, spec_SNR=spec_SNR,  
   >>>                      number_of_processors=number_of_processors)


Alternatively you can use the command line to prepare a sequence of days in one batch:

.. code-block:: bash

   >>> build_time_domain_snpcc.py -d 20 21 22 23 -p <path to raw data dir> 
   >>>        -o <path to output time domain dir> -q 2 -c True -nc 5

For PLASTiCC
^^^^^^^^^^^^

You can perform the entire analysis for one day of the survey using the `PLAsTiCCPhotometry class <https://resspect.readthedocs.io/en/latest/api/resspect.PLAsTiCCPhotometry.html>`_:


.. code-block:: python
   :linenos:

   >>> from resspect.time_domain_snpcc import PLAsTiCCPhotometry
   >>> from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES

   # required variables
   >>> create_daily_files = True               # create 1 file for each day of survey (do this only once!)
   >>> output_dir = '~/results/time_domain/'
   >>> raw_data_dir = '~/data/zenodo_dir/'     # path to PLAsTiCC zenodo files  
   
   # selected optional variables 
   >>> field = 'DDF'                           # DDF or WFD
   >>> get_cost = True                         # calculate cost of each observation
   >>> queryable_criteria = 2                  # if 2, estimate brightness at time of query
   >>> sample = 'test'                         # original plasticc sample
   >>> vol = 1                                 # index of plasticc zenodo file for test sample
   >>> spec_SNR = 10                           # minimum SNR required for spec follow-up
   >>> tel_names = ['4m, 8m']                  # name of telescopes considered for spec follow-up
   >>> tel_sizes = [4, 8]                      # size of primay mirrors, in m
   >>> time_window = [400, 401]                # days since the beginning of the survey to be processed
    
   # start PLAsTiCCPhotometry object
   >>> photom = PLAsTiCCPhotometry()
   >>> photom.build()
   
   # at first, create one file per day of the survey, this will creat empty files for [0, 1095]
   >>> if create_daily_files:
            photom.create_all_daily_files(output_dir=output_dir,
                                          get_cost=get_cost)

   # read metadata
   >>> photom.read_metadata(path_to_data_dir=raw_data_dir, 
                            classes=PLASTICC_TARGET_TYPES.keys(),
                            field=field, 
                            meta_data_file_name= 'plasticc_' + sample + '_metadata.csv.gz')
   
   # get all object ids
   >>> ids = photom.metadata['object_id'].values
    
   # For each light curve, feature extract days of the survey in "time_window"
   >>> for snid in ids:
            photom.fit_one_lc(raw_data_dir=raw_data_dir, snid=snid, 
                              output_dir=output_dir,
                              vol=vol, queryable_criteria=queryable_criteria,
                              get_cost=get_cost, 
                              tel_sizes=tel_sizes,
                              tel_names=tel_names, 
                              spec_SNR=spec_SNR, 
                              time_window=time_window, sample=sample)
                              
Alternatively you can use the command line to prepare a sequence of days in one batch:

.. code-block:: bash

   >>> build_time_domain_plasticc.py -df True -o <path to output dir> 
   >>>      -i <path to input zenodo dir> -ss DDF -g True -c 2 -s test -v 1
   >>>      -snr 10 -tw 400 401


.. warning::
   We show above a few of the parameters you can tune in this stage. 
   Please see docstring for  `PLAsTiCCPhotometry class <https://resspect.readthedocs.io/en/latest/api/resspect.PLAsTiCCPhotometry.html>`_ for more options regarding the feature extraction procedure, and 
   `exposure_time_calculator <https://resspect.readthedocs.io/en/latest/api/resspect.exposure_time_calculator.html>`_ to check what are the parameters used to estimate required exposure time in each telescope.

