# Copyright 2020 resspect software
# Author: The RESSPECT team
#         Initial skeleton taken from ActSNClass
#
# created on 02 March 2020
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

import numpy as np

from actsnclass.query_strategies import *

__all__ = ['percentile_sampling']


def percentile_sampling(class_prob: np.array, test_ids: np.array,
                  queryable_ids: np.array, perc: float) -> list:
    """Search for the sample at a specific percentile of uncertainty.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    perc: float in [0,1]
        Percentile used to identify obj to be queried.
  
    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
    """

    # calculate distance to the decision boundary - only binary classification
    dist = abs(class_prob[:, 1] - 0.5)

    # get indexes in increasing order
    order = dist.argsort()

    # get index of wished percentile
    perc_index = int(order.shape[0] * perc)

    # return the index of the highest object at the requested percentile
    return list([perc_index])


def main():
    return None


if __name__ == '__main__':
    main()
