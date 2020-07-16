.. _learnloop:


Active Learning loop
====================

Details on running 1 loop
-------------------------

Once the data has been pre-processed, analysis steps 2-4 can be performed directly using the ``DataBase`` object.

For start, we can load the feature information:

.. code-block:: python
   :linenos:

   >>> from resspect import DataBase

   >>> path_to_features_file = 'results/Bazin.dat'

   >>> data = DataBase()
   >>> data.load_features(path_to_features_file, method='Bazin', screen=True)
   Loaded  21284  samples!

Notice that this data has some pre-determine separation between training and test sample:

.. code-block:: python
   :linenos:

   >>> data.metadata['orig_sample'].unique()
   array(['test', 'train'], dtype=object)

You can choose to start your first iteration of the active learning loop from the original training sample
flagged int he file OR from scratch. As this is our first example, let's do the simple thing and start from the original
training sample. The code below build the respective samples and performs the classification:

.. code-block:: python
   :linenos:

   >>> data.build_samples(initial_training='original', nclass=2, screen=True)
   ** Inside build_orig_samples: **
      Training set size:  1093
      Test set size:  20191
      Validation set size:  20191
      Pool set size:  20191
         From which queryable:  20191 

   >>> data.classify(method='RandomForest')
   >>> data.classprob                        # check classification probabilities
   array([[0.461, 0.539],
          [0.346, 0.654],
          ...,
          [0.398, 0.602],
          [0.396, 0.604]])

.. hint:: If you wish to start from scratch, just set the `initial_training=N` where `N` is the number of objects in you want in the initial training. The code will then randomly select `N` objects from the entire sample as the initial training sample. It will also impose that at least half of them are SNe Ias.

For a binary classification, the  output from the classifier for each object (line) is presented as a pair of floats, the first column
corresponding to the probability of the given object being a Ia and the second column its complement.

Given the output from the classifier we can calculate the metric(s) of choice:

.. code-block:: python
   :linenos:

   >>> data.evaluate_classification(metric_label='snpcc')
   >>> print(data.metrics_list_names)           # check metric header
   ['acc', 'eff', 'pur', 'fom']

   >>> print(data.metrics_list_values)          # check metric values
   [0.5975434599574068, 0.9024767801857585,
   0.34684684684684686, 0.13572404702012383]


Running a number of iterations in sequence
------------------------------------------

We provide a function where all the above steps can be done in sequence for a number of iterations.
In interactive mode, you must define the required variables and use the :py:mod:`resspect.learn_loop` function:

.. code-block:: python
   :linenos:

   >>> from resspect.learn_loop import  learn_loop

   >>> nloops = 1000                                  # number of iterations
   >>> method = 'Bazin'                               # only option in v1.0
   >>> ml = 'RandomForest'                            # classifier
   >>> strategy = 'RandomSampling'                    # learning strategy
   >>> input_file = 'results/Bazin.dat'               # input features file
   >>> metric = 'results/metrics.dat'                 # output metrics file
   >>> queried = 'results/queried.dat'                # output query file
   >>> train = 'original'                             # initial training
   >>> batch = 1                                      # size of batch

   >>> learn_loop(nloops=nloops, features_method=method, classifier=ml,
   >>>            strategy=strategy, path_to_features=input_file, output_metrics_file=metric, 
   >>>            output_queried_file=queried, training=train, batch=batch)

Alternatively you can also run everything from the command line:

.. code-block:: bash

   >>> run_loop.py -i <input features file> -b <batch size> -n <number of loops>
   >>>             -m <output metrics file> -q <output queried sample file>
   >>>             -s <learning strategy> -t <choice of initial training>


Active Learning loop in time domain
===================================

Considering that you have previously prepared the time domain data, you can run the active learning loop
in its current form either by using the :py:mod:`resspect.time_domain_loop` module:

.. code-block:: python
    :linenos:

    >>> from resspect import time_domain_loop
    
    >>> days = [20, 180]
    >>> training = 'original'
    >>> strategy = 'UncSampling'
    >>> n_estimators = 1000
    >>> batch = 1
 
    >>> sep_files = True
    
    >>> output_diag_file = 'results/metrics_' + strategy + '_' + str(training) + \
                           '_batch' + str(batch) +  '.dat'
    >>> output_query_file = 'results/queried_' + strategy + '_' + str(training) + \
                            '_batch' + str(batch) +  '.dat'
    >>> path_to_features_dir = 'data/pool/'
  
    >>> batch = None # use budgets instead              # if int, ignore cost per observation
    >>> budgets = (6. * 3600, 6. * 3600)                # budget of 6 hours per night of observation
    >>> classifier = 'RandomForest'
    >>> clf_bootstrap = False 
    >>> feature_method = 'Bazin'
    >>> screen = True
    >>> fname_pattern = ['day_', '.dat']                # pattern on filename where different days of the survey are stored
    >>> canonical = False
    >>> queryable= True
    
    >>> path_to_ini_files = {}
    >>> path_to_ini_files['train'] = 'data/Train.csv'
    >>> path_to_ini_files['test'] = 'data/Test.csv'
    >>> path_to_ini_files['validation'] = 'data/Validation.csv'
    >>> survey='DES'
    
    >>> # run time domain loop
    >>> time_domain_loop(days=days, output_metrics_file=output_diag_file,
    >>>                  output_queried_file=output_query_file,
    >>>                  path_to_features_dir=path_to_features_dir,
    >>>                  budgets=budgets, clf_bootstrap=clf_bootstrap,
    >>>                  strategy=strategy, fname_pattern=fname_pattern, batch=batch, classifier=classifier,
    >>>                  canonical=canonical, sep_files=sep_files,
    >>>                  screen=screen, initial_training=training, path_to_ini_files=path_to_ini_files,
    >>>                  survey=survey, queryable=queryable, n_estimators=n_estimators)


Make sure you check the full documentation of the module to understand which variables are required depending
on the case you wish to run.

For example, to run with SNPCC data, the larges survey interval you can run is between 20 and 182 days,
the corresponding option will be `-d 20 182`.

In the example above, if you choose to start from the original training sample, `-t original` you must also
input the path to the file containing the full light curve analysis - so the full initial training can
be read. This option corresponds to `-t original -fl <path to full lc features>`.

More details can be found in the corresponding `docstring <https://github.com/COINtoolbox/resspect/blob/master/resspect/scripts/run_time_domain.py>`_.


Once you ran one or more options, you can use the :py:mod:`resspect.plot_results` module, as described in the :ref:`produce plots <plotting>` page.
The result will be something like the plot below (accounting for variations due to initial training).


.. image:: images/time_domain.png
   :align: center
   :height: 448 px
   :width: 640 px
   :alt: Example of time domain output.


.. warning:: At this point there is no `Canonical sample` option implemented for the time domain module.
