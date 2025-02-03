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

import argparse

from resspect.time_domain_snpcc import SNPCCPhotometry

__all__ = ['build_time_domain_snpcc']


def build_time_domain_snpcc(user_choice):
    """Generates features files for a list of days of the survey.

    Parameters
    ----------
    -c: bool
        If True, calculate cost of spectroscopy in each day.
    -d: sequence
        Sequence of days since the begin of the survey to be processed.
    -p: str
        Complete path to raw data directory.
    -q: int
        Queryable criteria.
    -o: str
        Complete path to output time domain directory.
    -f: str (optional)
        Feature method. Only 'Bazin' is accepted at the moment.
    -g: int (optional)
        Gap in days since last observation when the measured magnitude
        can be used for spectroscopic time estimation. Default is 2.
    -n: sequence (optional)
        Sequence with telescope names. Default is ["4m", "8m"].
    -snr: float (optional)
        SNR required for spectroscopic follow-up. Default is 10.
    -t: sequence (optional)
        Primary mirrors diameters of potential spectroscopic telescopes.
        Only used if "get_cost == True". Default is [4, 8].
    -nc: int (optional)
        Number of cores used in calculation. Default is 1.

    Examples
    -------
    Use it directly from the command line.

    >>> build_time_domain.py -d 20 21 22 23 -p <path to raw data dir> 
    >>>      -o <path to output time domain dir> -q 2 -c True -nc 10
    """
    path_to_data = user_choice.raw_data_dir
    output_dir = user_choice.output
    day = user_choice.day_of_survey
    get_cost = user_choice.get_cost
    queryable_criteria = user_choice.queryable_criteria
    feature_method = user_choice.feature_method
    days_since_obs = user_choice.days_since_obs
    tel_sizes = user_choice.tel_sizes
    tel_names = user_choice.tel_names
    spec_SNR = user_choice.spec_SNR
    number_of_processors = user_choice.n_cores

    for item in day:
        data = SNPCCPhotometry()
        data.create_daily_file(output_dir=output_dir, day=item, get_cost=get_cost)
        data.build_one_epoch(raw_data_dir=path_to_data, day_of_survey=int(item),
                             time_domain_dir=output_dir,
                             feature_extractor=feature_method, 
                             days_since_obs=days_since_obs,
                             queryable_criteria=queryable_criteria, 
                             get_cost=get_cost, tel_sizes=tel_sizes,
                             tel_names=tel_names, spec_SNR=spec_SNR,
                             number_of_processors=number_of_processors)


def main():
    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='resspect - '
                                                 'Prepare Time Domain module')

    parser.add_argument('-d', '--day', dest='day_of_survey', required=True,
                        nargs='+', help='Day after survey starts.')
    parser.add_argument('-p', '--path-to-data', required=True, type=str,
                        dest='raw_data_dir',
                        help='Complete path to raw data directory.')
    parser.add_argument('-q', '--queryable-criteria', required=False, type=int,
                        dest='queryable_criteria', default=1, 
                        help='Criteria to determine if an obj is queryable.\n' + \
                        '1 -> r-band cut on last measured photometric point.\n' + \
                        '2 -> if last obs was further than a given limit, ' + \
                        'use Bazin estimate of flux today. Otherwise, use' + \
                        'the last observed point. Default is 1.')
    parser.add_argument('-o', '--output', dest='output', required=True,
                        type=str, help='Path to output time domain directory.')
    parser.add_argument('-c', '--calculate-cost', dest='get_cost', default=False, 
                        help='Calculate cost of spectra in each day.')
    parser.add_argument('-f', '--feature-method', dest='feature_method', type=str,
                        required=False, default='bazin', help='Feature extraction method. ' + \
                        'Only "bazin" is accepted at the moment.')
    parser.add_argument('-g', '--days-since-obs', dest='days_since_obs', required=False,
                        type=int, default=2, help='Gap in days since last observation ' + \
                        'when the measured magnitude can be used for spectroscopic ' + \
                        'time estimation. Default is 2.')
    parser.add_argument('-t', '--telescope-sizes', dest='tel_sizes', required=False,
                        nargs='+', default=[4, 8], help='Primary mirrors diameters ' + \
                        'of potential spectroscopic telescopes. Only used if ' + \
                        '"get_cost == True". Default is [4, 8].')
    parser.add_argument('-n', '--telescope-names', dest='tel_names', required=False,
                        nargs='+', default=['4m', '8m'], help='Sequence of telescope ' + \
                        'names. Default is ["4m", "8m"].')
    parser.add_argument('-snr', '--spec-SNR', dest='spec_SNR', required=False,
                        default=10, help='SNR required for spectroscopic follow-up. ' + \
                        'Default is 10.')
    parser.add_argument('-nc', '--n-cores', dest='n_cores', required=False,
                        default=1, help='Number of cores used. ' + \
                        'Default is 1.', type=int)

    # get input directory and output file name from user
    from_user = parser.parse_args()

    build_time_domain_snpcc(from_user)


if __name__ == '__main__':
    main()