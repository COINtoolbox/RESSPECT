# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 14 April 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['run_time_domain']

import argparse

from resspect.time_domain_loop import time_domain_loop


def str2none(v):
    """Transform a given variable into None or str.

    Parameters
    ----------
    v: str or None or int

    Returns
    -------
    None or str or int
    """
    
    if v == None or v == 'original':
        return v
    elif str(v) == 'None' or str(v[0]) == 'None':
        return None
    elif isinstance(v, list):
        return ([int(v[0]), int(v[1])])
    elif int(str(v)) > 0:
        return int(v)

def run_time_domain(user_choice):
    """Command line interface to the Time Domain Active Learning scenario.

    Parameters
    ----------
    -d: sequence
         List of 2 elements. First and last day of observations since the
         beginning of the survey.
    -m: str
        Full path to output file to store metrics of each loop.
    -q: str
        Full path to output file to store the queried sample.
    -f: str
        Complete path to directory holding features files for all days.
    -s: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    -b: int or None (optional)
        If int,  size of batch to be queried in each loop. 
        If None, use budget of available telescope time to select batches.
        Default is 1.
    -bd: tuple or list of tuples (optional)
        Each element of the tuple represents available time in one spectroscopic telescope.
        Only used if -b == None. Default is None.
    -c: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees'
        'K-NNclassifier' and 'MLPclassifier' are implemented.
    -cn: bool (optional)
        If True, this concerns the canonical sample. Default is False.
    -fl: file containing full light curve features to train on
         if -t original is chosen.
    -fp: list of str (optional)
        List of strings containing pattern for file name with features for each day.
        Default is ['day_', '.csv']
    -it: str (optional)
        Path to initial training file. Only used if -sp == True.
        Default is None.
    -n: int (optional)
        Number of estimators (trees in the forest). 
        Only used if classifier == 'RandomForest'. Default is 1000.
    -pv: str (optional)
        Path to validation file. Only used if -sp == True.
        Default is False.
    -pt: str (optional)
        Path to test file. Only used if -sp == True.
        Default is None.
    -qb: bool (optional)
        If True, consider the mag of the object at the moment of query.
        Default is True.
    -sc: bool (optional)
        If True, display comment with size of samples on screen.
    -sf: bool (optional)
        If True, it assumes samples are stored in separate files. 
        Default is False.
    -sv: str (optional)
        Survey. Options are 'DES' or 'LSST'. Default is 'DES'.
    -t: str or int
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.

    Returns
    -------
    metric file: file
        File with metrics calculated in each iteration.
    queried file: file
        All objects queried during the search, in sequence.

    Examples
    --------
    Use directly from the command line.

    >>> run_time_domain.py -d <first day of survey> <last day of survey>
    >>>        -m <output metrics file> -q <output queried file> -f <features directory>
    >>>        -s <learning strategy> -t <training choice>
    >>>        -fl <path to full light curve features >

    Be aware to check the default options as well!
    """
    """
    # set parameters"""
    days = [int(user_choice.days[0]), int(user_choice.days[1])]
    output_metrics_file = user_choice.metrics
    output_query_file = user_choice.queried
    path_to_features_dir = user_choice.features_dir
    strategy = user_choice.strategy
    training = str2none(user_choice.training)
    queryable = user_choice.queryable

    
    batch = str2none(user_choice.batch)
    classifier = user_choice.classifier
    screen = user_choice.screen
    fname_pattern = user_choice.fname_pattern
    n_estimators = user_choice.n_estimators
    
    sep_files = user_choice.sep_files
    budgets = str2none(user_choice.budgets)
    survey = user_choice.survey
    canonical = user_choice.canonical
    
    path_to_ini_files = {}
    path_to_ini_files['train'] = user_choice.full_features
    path_to_ini_files['validation'] = user_choice.val
    path_to_ini_files['test'] = user_choice.test    
    
    # run time domain loop
    time_domain_loop(days=days, output_metrics_file=output_metrics_file,
                 output_queried_file=output_query_file, 
                 path_to_ini_files=path_to_ini_files,
                 path_to_features_dir=path_to_features_dir,
                 strategy=strategy, fname_pattern=fname_pattern, 
                 batch=batch, classifier=classifier,
                 sep_files=sep_files, budgets=budgets,
                 screen=screen, initial_training=training,
                 survey=survey, queryable=queryable, n_estimators=n_estimators,
                 canonical=canonical)


def str2bool(v):
    """Convert str into bool.

    Parameters
    ----------
    v: str or bool

    Returns
    -------
    bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='resspect - '
                                                 'Time Domain loop module')

    parser.add_argument('-d', '--days', dest='days', required=True,
                        help='First and last day of survey.',
                        nargs='+')
    parser.add_argument('-m', '--metrics', dest='metrics', required=True,
                        help='File to store metrics results.')
    parser.add_argument('-q', '--queried', dest='queried',
                        help='File to store queried sample.',
                        required=True)
    parser.add_argument('-f', '--feat-dir', dest='features_dir',
                        required=True, help='Path to directory '
                                            'of feature files for each day.')
    parser.add_argument('-s', '--strategy', dest='strategy',
                        required=True,
                        help='Active Learning strategies. Possibilities are'
                             '"RandomSampling" or UncSampling.')
    parser.add_argument('-b', '--batch', dest='batch', required=False,
                        help='Number of queries in each iteration.',
                        default=1)
    parser.add_argument('-c', '--classifier', dest='classifier',
                        required=False, default='RandomForest',
                        help='Classifier. Currently only accepts '
                             '"RandomForest", "GradientBoostedTrees",'
                             ' "K-NNclassifier" and "MLPclassifier".')
    parser.add_argument('-fl', '--full-light-curve',
                        dest='full_features', required=False,
                        default=' ', help='Path to full light curve features for initial training.')
    parser.add_argument('-sc', '--screen', dest='screen', required=False,
                        default=True, type=str2bool,
                        help='If True, display size info on training and '
                             'test sample on screen.')
    parser.add_argument('-t', '--training', dest='training', required=False,
                        default='original', help='Choice of initial training'
                                                 'sample. It can be "original"'
                                                 'or an integer.')
    parser.add_argument('-sf', '--sep-files', dest='sep_files', required=False, type=str2bool,
                       default=False, help='If True, assumes samples are given in separate files.' + \
                       ' Default is False.')
    parser.add_argument('-bd', '--budgets', dest='budgets', required=False, default=None,
                       help='List of available telescope resources for querying.', nargs='+')
    parser.add_argument('-pv', '--path-validation', dest='val', required=False, default=None,
                       help='Path to validation sample. Only used if sep_files=True.')
    parser.add_argument('-pt', '--path-test', dest='test', required=False, default=None,
                       help='Path to test sample. Only used if sep_files == True.')
    parser.add_argument('-fp', '--fname-pattern', dest='fname_pattern', required=False, nargs='+', 
                        default=['day_', '.csv'],
                        help='Filename pattern for time domain.')
    parser.add_argument('-sv', '--survey', dest='survey', required=False, default='DES',
                       help='Survey. Options are "DES" or "LSST". Default is "DES".')
    parser.add_argument('-qb', '--queryable', dest='queryable', required=False, default=True,
                        type=str2bool,
                       help='If True consider mag of the object at the moment of query.')
    parser.add_argument('-n', '--n-estimators', dest='n_estimators', required=False, default=1000, 
                       help='Number of trees in Random Forest.', type=int)
    parser.add_argument('-cn', '--canonical', dest='canonical', required=False, default=False,
                        type=str2bool,
                       help='If True this concerns the canonical sample. Default is False.')
    
    from_user = parser.parse_args()

    run_time_domain(from_user)
  

if __name__ == '__main__':
    main()
