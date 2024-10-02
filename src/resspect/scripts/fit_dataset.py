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

from resspect.fit_lightcurves import fit_snpcc
from resspect.fit_lightcurves import fit_plasticc

__all__ = ['fit_dataset']


def fit_dataset(user_choices):
    """Fit the entire sample with the Bazin function.

    All results are saved to file.

    Parameters
    ----------
    -o: str
        Path to output feature file.
    -s: str
        Simulation name. Options are 'SNPCC', 'RESSPECT' or 'PLAsTiCC'.
    -dd: str (optional)
        Path to directory containing raw data.
        Only used for SNPCC simulations.
    -hd: str (optional)
        Path to header file. Only used for PLASTICC simulations.
    -p: str (optional)
        Path to photometry file. Only used for PLASTICC simulations.
    -sp: str or None (optional)
        Sample to be fitted. Options are 'train', 'test' or None.
        Default is None.
    -n: int or None (optional)
        Number of cores to be used. If None all cores are used. 
        Default is 1.
    -f: str (optional)
        Function used for feature extraction. Options are "bazin" and "bump".
        Default is "bazin". Only used for SNPCC for now.
        
    Examples
    --------

    For SNPCC: 

    >>> fit_dataset.py -s SNPCC -dd <path_to_data_dir> -o <output_file>

    For PLAsTiCC:

    >>> fit_dataset.py -s <dataset_name> -p <path_to_photo_file> 
             -hd <path_to_header_file> -o <output_file> 
    """

    # raw data directory
    data_dir = user_choices.input
    features_file = user_choices.output
    ncores = user_choices.ncores

    if user_choices.sim_name.lower() == 'snpcc':
        # fit the entire sample
        fit_snpcc(path_to_data_dir=data_dir, features_file=features_file,
                  number_of_processors=ncores, feature_extractor=user_choices.function)

    elif user_choices.sim_name.lower() == 'plasticc':
        fit_plasticc(path_photo_file=user_choices.photo_file,
                    path_header_file=user_choices.header_file,
                    output_file=features_file,
                    sample=user_choices.sample,
                    number_of_processors=ncores)

    else:
        raise ValueError("-s or --simulation not recognized. Options are 'SNPCC' or 'PLAsTiCC'.")

def main():

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='resspect - Fit Light curves module')
   
    parser.add_argument('-dd', '--datadir', dest='input',
                        help='Path to directory holding raw data. Only used for SNPCC',
                        required=False, default=' ')
    parser.add_argument('-hd', '--header', dest='header_file', 
                        help='Path to header file. Only used for PLASTICC.',
                        required=False, default=' ')
    parser.add_argument('-o', '--output', dest='output', help='Path to output file.', 
                        required=True)
    parser.add_argument('-p', '--photo', dest='photo_file',
                        help='Path to photometry file. Only used for PLASTICC.',
                        required=False, default=' ')
    parser.add_argument('-s', '--simulation', dest='sim_name', 
                        help='Name of simulation (data set). ' + \
                             'Options are "SNPCC" or "PLAsTiCC".',
                        required=True)
    parser.add_argument('-sp', '--sample', dest='sample',
                        help='Sample to be fitted. Options are "train", ' + \
                             ' "test" or None.',
                        required=False, default=None)
    parser.add_argument('-n', '--number-of-processors', dest='ncores', 
                       help='Number of processors. Default is 1.',
                       required=False, default=1)
    parser.add_argument('-f', '--function', dest='function', 
                       help='Function used for feature extraction.',
                       required=False, default="bazin")

    user_input = parser.parse_args()

    fit_dataset(user_input)


if __name__ == '__main__':
    main()