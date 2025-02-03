# Copyright 2020 resspect software
# Author: Kara Ponder and Emille E. O. Ishida
#
# created on 21 April 2020
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
import numpy as np

from resspect.cosmo_metric_utils import compare_two_fishers
from resspect.cosmo_metric_utils import fisher_results
from resspect.cosmo_metric_utils import update_matrix

__all__ = ['calculate_cosmology_metric']

def calculate_cosmology_metric(args):
    """Calculate Fisher matrix-based cosmology metric.

    Parameters
    ----------
    -c: str
        Path to second dataset to compare.
    -d: str
        Data to calculate fisher matrix (ID, z, mu, err).
    -N: int
        N number of top choices for follow-up.
    -u: str
        Path to data to be used to update the fisher matrix.
    -o: str (optional)
        Output file name. Stores cosmology metric results.
        Default is 'cosmology_metric_output.dat'.
    -s: bool (optional)
        Sort the output by how much uncertainty reduced.
        Default is False.
    -sc: bool (optional)
        Print intermediate steps to screen.
        Default is False.
    -t: bool (optional)
        Print the total uncertainty update if all objects included.
        Default is False.
    -ns: bool (optional)
        Print the total uncertainty update if all objects included.
        Default is True. 

    Examples
    --------

    >>> calculate_cosmology_metric.py -d <path_to_first_dataset> 
            -c <path_to_second_dataset> -N 1 -u <path_to_update_matrix>
           -t True -t True -ns True
    """


    # read distances for first set of photometrically classified Ia
    original_data =  np.loadtxt(args.data, unpack=True)

    if args.comparison_data:

        # read distances for comparison 
        data2  =  np.loadtxt(args.comparison_data, unpack=True)

        ctf = compare_two_fishers(original_data, data2, screen=False)

        if args.screen:
            print('Difference between 2 Fisher Matrices:', ctf)
            print('\nif positive, data set 2 has tighter constraints than data set 1.')
            print('if negative, data set 1 has tighter constraints than data set 2.')

    if args.update_data:

        # update initial data_set
        new_data = pd.read_csv(args.update_data, index_col=False)

        sigma, covmat = fisher_results(original_data[1], original_data[3])

        try:
            if np.shape(new_data)[0] > 1:
                u = np.array([update_matrix(up[0],
                                            up[1],
                                            covmat)[1][1]
                              for up in zip(new_data['z'], new_data['err'])])
                new_data['Update'] = u

                if args.total_update:
                    print('Change in w0', u.sum())

                if args.sort_update:
                    new_data.sort_values(by='Update', ascending=False, inplace=True)

                    if args.N is not None:
                        new_data = new_data.head(n=args.N)

                if args.save_file:
                    new_data.to_csv(args.output, index=False)

            else:
                print('Change in w0', update_matrix(new_data['z'],
                                                    new_data['err'],
                                                    covmat)[1][1])

        except:
            print('Something went wrong...')

def main():

    usage = "usage: [%prog] [options]"

    parser = argparse.ArgumentParser(description='resspect - Calculate cosmology metric.')

    parser.add_argument('-c', '--comp-data', dest='comparison_data', type=str,
                        help='Second dataset to compare.')

    parser.add_argument('-d', '--data', dest='data', type=str,
                        help='Data to calculate fisher matrix (ID, z, mu, err).')

    parser.add_argument('-N', '-Nchoices', dest='N', type=int,
                        default=None,
                        help='N number of top choices for follow-up.')
    
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='cosmology_metric_output.dat',
                        help='File name for output.')

    parser.add_argument('-s', '--sorted', dest='sort_update',
                        action='store_true',
                        default=False,
                        help='Sort the output by how much uncertainty reduced.')

    parser.add_argument('-sc', '--screen', dest='screen', default=False,
                        help='Print intermediate steps to screen.')

    parser.add_argument('-t', '--total-uncertainty', dest='total_update',
                        action='store_true',
                        default=False,
                        help='Print the total uncertainty update if all objects included.')

    parser.add_argument('-u', '--update-data', dest='update_data', type=str,
                        help='Data to use to update the fisher matrix.')

    parser.add_argument('-ns', '--no-save', dest='save_file',
                        action='store_false',
                        default=True,
                        help='Print the total uncertainty update if all objects included.')

    args = parser.parse_args()

    calculate_cosmology_metric(args)

if __name__ == '__main__':
    main()