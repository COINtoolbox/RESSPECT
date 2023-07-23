.. _preprocessing:

Feature Extraction
==================

The first stage in consists in transforming the raw data into a uniform data matrix which will subsequently be given
as input to the learning algorithm.

``resspect`` can handle csv data from the Photometric LSST Astronomical Classification Challenge (`PLAsTiCC <https://zenodo.org/record/2539456#.Xrsk33UzZuQ>`_)  and text-like data from the SuperNova Photometric Classification Challenge (`SNPCC <https://arxiv.org/abs/1008.1024>`_).

For PLAsTiCC:
^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   >>> from resspect import fit_plasticc_bazin

   >>> photo_file = '~/plasticc_train_lightcurves.csv' 
   >>> header_file = '~/plasticc_train_metadata.csv.gz'
   >>> output_file = 'results/PLAsTiCC_Bazin_train.dat'            

   >>> sample = 'train'

   >>> fit_plasticc_bazin(photo_file, header_file, output_file, sample=sample)


For SNPCC:
^^^^^^^^^^

.. code-block:: python
   :linenos:

   >>> from resspect import fit_snpcc_bazin

   >>> path_to_data_dir = 'data/SIMGEN_PUBLIC_DES/'            # raw data directory
   >>> output_file = 'results/Bazin.dat'                              # output file

   >>> fit_snpcc_bazin(path_to_data_dir=path_to_data_dir, features_file=output_file)



The same result can be achieved using the command line:

.. code-block:: bash
    :linenos:

    # for PLAsTiCC
    >>> fit_dataset.py -s <dataset_name> -p <path_to_photo_file> 
             -hd <path_to_header_file> -sp <sample> -o <output_file> 

    # for SNPCC
    >>> fit_dataset.py -s SNPCC -dd <path_to_data_dir> -o <output_file>
