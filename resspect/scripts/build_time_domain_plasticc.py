# Copyright 2022 resspect software
# Author: Emille E. O. Ishida
#
# created on 18 January 2022
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

import argparse

from resspect.time_domain_plasticc import PLAsTiCCPhotometry
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES

__all__ = ['build_time_domain_plasticc']


def build_time_domain_plasticc(user_choice):
    """Generates features files for all objects in a metadata file.
    
    Parameters
    ----------
    -i: str
        Input PLAsTiCC zenodo directory.
    -o: str
        Output directory.
    -c: int (optional)
        queryable_criteria: int [1, 2 or 3] 
        Criteria to determine if an obj is queryable.
        1 -> Cut on last measured photometric point.
        2 -> if last obs was further than days_since_last_obs,
             use Bazin estimate for today. Otherwise, use
             the last observed point.
        Default is 1.
    -d: int (optional)
        days_since_last_obs
        If there is an observation within these days, use the
        measured value, otherwise estimate current mag.
        Only used if "criteria == 2". Default is 2.
    -df: bool (optional)
        If True, create new daily files for the chosen time windown.
        Default is False.
    -f: str (optional)
        Feature extraction method.
        Only possibility is 'Bazin'.
    -g: bool (optional)
        get_cost
        If True, calculate cost of taking a spectra in the last
        observed photometric point. Default is False.
    -v: int (optional)
        Index of photometry file [1,2,3,..,11]. 
        Only used if sample == 'test'. Default is 1.
    -s: str (optinal)
        sample
        Sample to load, 'train' or 'test'. Default is 'test'.
    -snr: float (optional)
        SNR required for spectroscopic follow-up. Default is 10.
    -ss: str or None (optional)
        Telescope cadence.
       'DDF', 'WFD' or 'DDF+WFD'. Default is 'DDF+WFD'.
    -tn: list (optional)
        tel_names
        Names of the telescopes under consideration for spectroscopy.
        Only used if "get_cost == True".
        Default is ["4m", "8m"].
    -ts: list (optional)
        tel_sizes
        Primary mirrors diameters of potential spectroscopic telescopes.
        Only used if "get_cost == True".
        Default is [4, 8].
    -tw: list (optional)
        time_window
        Days of the survey to process, in days since the start of the survey. 
        Default is the entire survey = [0, 1095].
        
    Examples
    -------
    Use it directly from the command line.

    >>> build_time_domain_PLAsTiCC.py -o <path to output dir> 
    >>>      -i <path to input zenodo dir> 
    """
    
    create_daily_files = user_choice.create_daily_files
    output_dir = user_choice.output_dir
    raw_data_dir = user_choice.raw_data_dir
    
    days_since_last_obs = user_choice.days_since_last_obs
    feature_method = user_choice.feature_method
    field = user_choice.field
    get_cost = user_choice.get_cost
    queryable_criteria = user_choice.queryable_criteria
    sample = user_choice.sample
    spec_SNR = user_choice.spec_SNR
    tel_names = user_choice.tel_names
    tel_sizes = user_choice.tel_sizes
    time_window = [int(user_choices.time_window[0]),
                   int(user_choices.time_window[1])]
    vol = user_choice.vol
    
    photom = PLAsTiCCPhotometry()
    photom.build()
    
    print('output_dir = ', output_dir)
    print('create_daily_files = ', create_daily_files)
    print('raw_data_dir = ', raw_data_dir)
    print('field = ', field)
    print('get_cost = ', get_cost)
    print('queryable_criteria = ', queryable_criteria)
    print('sample = ', sample)
    print('spec_SNR = ', spec_SNR)
    print('tel_names = ', tel_names)
    print('tel_sizes = ', tel_sizes)
    print('time_window = ', time_window)
    print('vol = ', vol)
    return 0
    
    if create_daily_files:
        print('inside create_daily_files')
        photom.create_all_daily_files(output_dir=output_dir,
                                      get_cost=get_cost)
    
    photom.read_metadata(path_to_data_dir=raw_data_dir, 
                         classes=PLASTICC_TARGET_TYPES.keys(),
                         field=field, 
                         meta_data_file_name= 'plasticc_' + sample + '_metadata.csv.gz')
    
    ids = photom.metadata['object_id'].values
    
    for snid in ids:
        photom.fit_one_lc(raw_data_dir=raw_data_dir, snid=snid, 
                          output_dir=output_dir,
                          vol=vol, day=None, queryable_criteria=queryable_criteria,
                          days_since_last_obs=days_since_last_obs, get_cost=get_cost, 
                          tel_sizes=tel_sizes,
                          tel_names=tel_names, feature_method=feature_method, 
                          spec_SNR=spec_SNR, 
                          time_window=time_window, sample=sample)
    
def main():
    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='resspect - '
                                                 'Prepare Time Domain PLAsTiCC module')

    parser.add_argument('-df', '--create-daily-files', 
                        dest='create_daily_files', default=False, 
                        help='If True, create new daily files.')    
    parser.add_argument('-i', '--raw-data-dir', required=True, type=str,
                        dest='raw_data_dir',
                        help='Complete path to PLAsTiCC zenodo directory.')
    parser.add_argument('-c', '--queryable-criteria', required=False, type=int,
                        dest='queryable_criteria', default=1, 
                        help='Criteria to determine if an obj is queryable.\n' + \
                        '1 -> r-band cut on last measured photometric point.\n' + \
                        '2 -> if last obs was further than a given limit, ' + \
                        'use Bazin estimate of flux today. Otherwise, use' + \
                        'the last observed point. Default is 1.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        type=str, help='Path to output directory.')
    parser.add_argument('-d', '--days-since-obs', dest='days_since_last_obs', required=False,
                        type=int, default=2, help='Gap in days since last observation ' + \
                        'when the measured magnitude can be used for spectroscopic ' + \
                        'time estimation. Default is 2.')
    parser.add_argument('-f', '--feature-method', dest='feature_method', type=str,
                        required=False, default='Bazin', help='Feature extraction method. ' + \
                        'Only "Bazin" is accepted at the moment.')    
    parser.add_argument('-g', '--get-cost', dest='get_cost', 
                        default=False, 
                        help='Calculate cost of spectra in each day. Default is False.')
    parser.add_argument('-v', '--volume', dest='vol', type=int, required=False,
                       default=1, help='Index of photometric file. Only used if ' +\
                       'sample == test. Default is 1.')
    parser.add_argument('-s', '--sample', dest='sample', type=str, required=False,
                       default='test', help='Original sample: test or train. Default is test.')
    parser.add_argument('-snr', '--spec-SNR', dest='spec_SNR', required=False,
                        default=10, help='SNR required for spectroscopic follow-up. ' + \
                        'Default is 10.')
    parser.add_argument('-ss', '--survey-strategy', dest='field', type=str, required=False,
                       default=None, help='Survey strategy: DDF, WFD or DDF+WFD. Default is' +\
                        'DDF + WFD.')
    parser.add_argument('-ts', '--telescope-sizes', dest='tel_sizes', required=False,
                        nargs='+', default=[4, 8], help='Primary mirrors diameters ' + \
                        'of potential spectroscopic telescopes. Only used if ' + \
                        '"get_cost == True". Default is [4, 8].')
    parser.add_argument('-tn', '--telescope-names', dest='tel_names', required=False,
                        nargs='+', default=['4m', '8m'], help='Sequence of telescope ' + \
                        'names. Default is ["4m", "8m"].')
    parser.add_argument('-tw', '--time-windown', dest='time_window', required=False,
                       nargs='+', default=[0, 1095], help='List of first and last day '+\
                       'since the start of the survey to be fitted. Default is [0, 1095].')

    # get input directory and output file name from user
    user_choices = parser.parse_args()

    build_time_domain_plasticc(user_choices)


if __name__ == '__main__':
    main()