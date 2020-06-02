# Copyright 2020 resspect software
# Author: The RESSPECT team
#         Based on initial prototype developed by the CRP #4 team
#
# created on 2 March 2020
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

__all__ = ['knn']

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, **kwargs):
    """Random Forest classifier.
    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Features from sample to be classified.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.
    Returns
    -------
    predictions: np.array
        Predicted classes for test sample.
    prob: np.array
        Classification probability for test sample [pIa, pnon-Ia].
    """

    # create classifier instance
    clf = RandomForestClassifier(**kwargs)

    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob

def knn(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, nneighbors=10):
    """k-Nearest Neighbor classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    nneighbors: int (optional)
        Number of nearest neighbors to consider. 
        Default is 10.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    # create classifier instance
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob


def main():
    return None


if __name__ == '__main__':
    main()
