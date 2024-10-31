# Copyright 2020 resspect software
# Author: The RESSPECT team
#
# created on 14 April 2020
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['uncertainty_sampling',
           'random_sampling',
           'uncertainty_sampling_entropy',
           'uncertainty_sampling_least_confident',
           'uncertainty_sampling_margin',
           'qbd_mi',
           'qbd_entropy']

import numpy as np
import pandas as pd

QUERY_STRATEGY_REGISTRY = {}

class QueryStrategy:
    """Base class that all built-in query strategies inherit from.
    """
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int = 1,
                 query_threshold: float = 1.0,
                 screen: bool = False,
                 **kwargs):
        """Base initializer for all query strategies.

        Parameters
        ----------
        queryable_ids : np.array
            Set of ids for objects available for querying.
        test_ids : np.array
            Set of ids for objects in the test sample.
        batch : int, optional
            Number of objects to be chosen in each batch query, by default 1
        query_threshold : float, optional
            Threshold where a query is considered worth it, by default 1.0 (no limit)
        screen : bool, optional
            If True display on screen the shift in index and
            the difference in estimated probabilities of being Ia
            caused by constraints on the sample available for querying, by default False
        """

        self.queryable_ids = queryable_ids
        self.test_ids = test_ids
        self.batch = batch
        self.query_threshold = query_threshold
        self.screen = screen
        self.requires_ensemble = False

    def __init_subclass__(cls):
        """Register all subclasses of QueryStrategy in the QUERY_STRATEGY_REGISTRY."""
        if cls.__name__ in QUERY_STRATEGY_REGISTRY:
            raise ValueError(f"Duplicate classifier name: {cls.__name__}")

        QUERY_STRATEGY_REGISTRY[cls.__name__] = cls

    def sample(self, probability: np.array) -> list:
        """Abstract method that all subclasses must implement.

        Parameters
        ----------
        probability : np.array
            Classification probability. The shape of the input probability depends
            on the query strategy implementation. See the subclasses implementation
            of this method for more details.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried. If there are less queryable objects than the
            required batch it will return only the available objects
            -- so the list of objects to query can be smaller than 'self.batch'.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError


class UncSampling(QueryStrategy):
    """RESSPECT-specific implementation of uncertainty sampling."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest uncertainty in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability. One value per class per object.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return uncertainty_sampling(probability,
                                    test_ids=self.test_ids,
                                    queryable_ids=self.queryable_ids,
                                    batch=self.batch,
                                    screen=self.screen,
                                    query_thre=self.query_threshold)


class UncSampleEntropy(QueryStrategy):
    """RESSPECT-specific implementation of uncertainty sampling defined by entropy."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest uncertainty, defined by entropy,
        in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability. One value per class per object.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return uncertainty_sampling_entropy(probability,
                                            test_ids=self.test_ids,
                                            queryable_ids=self.queryable_ids,
                                            batch=self.batch,
                                            screen=self.screen,
                                            query_thre=self.query_threshold)


class UncSampleLeastConfident(QueryStrategy):
    """RESSPECT-specific implementation of uncertainty sampling defined by least confident."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest uncertainty, defined by least
        confident, in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability. One value per class per object.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return uncertainty_sampling_least_confident(probability,
                                                    test_ids=self.test_ids,
                                                    queryable_ids=self.queryable_ids,
                                                    batch=self.batch,
                                                    screen=self.screen,
                                                    query_thre=self.query_threshold)


class UncSampleMargin(QueryStrategy):
    """RESSPECT-specific implementation of uncertainty sampling defined by least confident."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest uncertainty, defined by max margin,
        in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability. One value per class per object.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return uncertainty_sampling_margin(probability,
                                           test_ids=self.test_ids,
                                           queryable_ids=self.queryable_ids,
                                           batch=self.batch,
                                           screen=self.screen,
                                           query_thre=self.query_threshold)


class RandomSampling(QueryStrategy):
    """RESSPECT-specific implementation of random sampling."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 queryable: bool = False,
                 **kwargs):
        """Randomly choose an object from the test sample.

        Parameters
        ----------
        queryable_ids : np.array
            Set of ids for objects available for querying.
        test_ids : np.array
            Set of ids for objects in the test sample.
        batch : int, optional
            Number of objects to be chosen in each batch query, by default 1
        query_threshold : float, optional
            Threshold where a query is considered worth it, by default 1.0 (no limit)
        screen : bool, optional
            If True display on screen the shift in index and
            the difference in estimated probabilities of being Ia
            caused by constraints on the sample available for querying, by default False
        queryable : bool, optional
            If True, check if randomly chosen object is queryable, by default False.
        """
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)
        self.queryable = queryable

    def sample(self, probability: np.array = None) -> list:
        """Randomly choose an object from the test sample.

        Parameters
        ----------
        probability : np.array
            Unused in this implementation.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried. If there are less queryable objects than the
            required batch it will return only the available objects
            -- so the list of objects to query can be smaller than 'batch'.
        """
        return random_sampling(test_ids=self.test_ids,
                               queryable_ids=self.queryable_ids,
                               batch=self.batch,
                               queryable=self.queryable,
                               query_thre=self.query_threshold,
                               screen=self.screen)


class QBDMI(QueryStrategy):
    """RESSPECT-specific implementation of QBDMI Strategy."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)
        self.requires_ensemble = True

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest uncertainty in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability from each model in the ensemble.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return qbd_mi(probability,
                      test_ids=self.test_ids,
                      queryable_ids=self.queryable_ids,
                      batch=self.batch,
                      screen=self.screen,
                      query_thre=self.query_threshold)


class QBDEntropy(QueryStrategy):
    """RESSPECT-specific implementation of QBDEntropy Strategy."""
    def __init__(self,
                 queryable_ids: np.array,
                 test_ids: np.array,
                 batch: int,
                 query_threshold: float,
                 screen: bool,
                 **kwargs):
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)
        self.requires_ensemble = True

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest entropy from the average predictions
        of the classifier ensemble. These can be instances where the classifiers
        agree (but are uncertain about the class) or disagree.

        Parameters
        ----------
        probability : np.array
            Classification probability from each model in the ensemble.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """
        return qbd_entropy(probability,
                           test_ids=self.test_ids,
                           queryable_ids=self.queryable_ids,
                           batch=self.batch,
                           screen=self.screen,
                           query_thre=self.query_threshold)

def compute_entropy(ps: np.array):
    """
    Calcualte the entropy for discrete distributoons assuming the events are
    indexed by the last dimension.

    Parameters
    ----------
    ps: np.array
        Probability disburtions to compute entropy of.

    Returns
    -------
    entropy: np.array
        Predicted classes.
    """
    return -1*np.sum(ps*np.log(ps + 1e-12), axis=-1)


def compute_qbd_mi_entropy(ensemble_probs: np.array):
    """
    Calculate the entropy of the average distribution from an ensemble of
    distributions. Calculate the mutual information between the members in the
    ensemble and the average distribution.

    Parameters
    ----------
    ensemble_probs: np.array
        Probability from ensembles where the first dimension is number of unique
        points, the second dimension is the number of ensemble members and the
        third dimension is the number of events.

    Returns
    -------
    entropy: np.array
    mutual information: np.array
    """
    avg_dist = np.mean(ensemble_probs, axis=1)
    entropy_avg_dist = compute_entropy(avg_dist)
    conditional_entropy = compute_entropy(ensemble_probs)
    mutual_information = entropy_avg_dist - np.mean(conditional_entropy, axis=1)
    return entropy_avg_dist, mutual_information


def uncertainty_sampling(class_prob: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty in predicted class.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """

    if class_prob.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # calculate distance to the decision boundary - only binary classification
    dist = abs(class_prob[:, 1] - 0.5)

    # get indexes in increasing order
    order = dist.argsort()

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)

    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('\n Inside UncSampling: ')
            print('       query_ids: ', test_ids[final_order][:batch], '\n')
            print('   number of test_ids: ', test_ids.shape[0])
            print('   number of queryable_ids: ', len(queryable_ids), '\n')
            print('   *** Displacement caused by constraints on query****')
            print('   0 -> ', list(order).index(final_order[0]))
            print('   ', class_prob[order[0]], '-- > ', class_prob[final_order[0]], '\n')

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])


def random_sampling(test_ids: np.array, queryable_ids: np.array,
                    batch=1, queryable=False, query_thre=1.0, seed=42,
                    screen=False) -> list:
    """Randomly choose an object from the test sample.

    Parameters
    ----------
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    queryable: bool (optional)
        If True, check if randomly chosen object is queryable.
        Default is False.
    query_thre: float (optinal)
        Threshold where a query is considered worth it.
        Default is 1.0 (no limit).
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    seed: int (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried. If there are less queryable objects than the
            required batch it will return only the available objects
            -- so the list of objects to query can be smaller than 'batch'.
    """

    # randomly select indexes to be queried
    np.random.seed(seed)
    indx = np.random.choice(np.arange(0, len(test_ids)), 
                            size=len(test_ids),
                            replace=False)

    if queryable:
        
        # flag only the queryable objects
        flag = list(pd.Series(data=test_ids[indx]).isin(queryable_ids))

        ini_index = flag.index(True)

        flag = np.array(flag)

        # check if there are queryable objects within threshold
        indx_query = int(len(flag) * query_thre)

        if sum(flag[:indx_query]) > 0:       
            if screen:
                print('\n Inside RandomSampling: ')
                print('       query_ids: ', test_ids[indx[flag]][:batch], '\n')
                print('   number of test_ids: ', test_ids.shape[0])
                print('   number of queryable_ids: ', len(queryable_ids), '\n')
                print('   inedex of queried ids: ', indx[flag][:batch])

            # return the corresponding batch size
            return list(indx[flag])[:batch]
        else:
            # return empty list
            return list([])
    else:
        return list(indx)[:batch]


def uncertainty_sampling_entropy(class_prob: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty, defined by entropy, in predicted class.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """
    if class_prob.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # calculate entropy
    entropies = (-1*np.sum(class_prob * np.log(class_prob + 1e-12), axis=1))

    # get indexes in increasing order
    order = entropies.argsort()[::-1]

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(class_prob[order[0]], '-- > ', class_prob[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])

def uncertainty_sampling_least_confident(class_prob: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty, defined by least confident, in predicted class.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """
    if class_prob.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # Get probability of predicted class
    prob_predicted_class = class_prob.max(axis=1)

    # get indexes in increasing order
    order = prob_predicted_class.argsort()

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(class_prob[order[0]], '-- > ', class_prob[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])

def uncertainty_sampling_margin(class_prob: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty, defined by max margin, in predicted class.

    Parameters
    ----------
    class_prob: np.array
        Classification probability. One value per class per object.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """
    if class_prob.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # Calculate margin between highest predicted class and second highest
    sorted_probs = np.sort(class_prob, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    # get indexes in increasing order
    order = margin.argsort()

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(class_prob[order[0]], '-- > ', class_prob[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])


def qbd_mi(ensemble_probs: np.array, test_ids: np.array,
                         queryable_ids: np.array, batch=1,
                         screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest uncertainty in predicted class.

    Parameters
    ----------
    ensemble_probs: np.array
        Classification probability from each model in the ensemble.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """
    if ensemble_probs.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # calculate distance to the decision boundary - only binary classification
    _, mis = compute_qbd_mi_entropy(ensemble_probs)

    # get indexes in decreasing order
    order = mis.argsort()[::-1]

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(ensemble_probs[order[0]], '-- > ', ensemble_probs[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])


def qbd_entropy(ensemble_probs: np.array, test_ids: np.array,
                queryable_ids: np.array, batch=1,
                screen=False, query_thre=1.0) -> list:
    """Search for the sample with highest entropy from the average predictions
    of the ensembled classifiers. These can be instances where the classifiers
    agree (but are uncertain about the class) or disagree.

    Parameters
    ----------
    ensemble_probs: np.array
        Classification probability from each model in the ensemble.
    test_ids: np.array
        Set of ids for objects in the test sample.
    queryable_ids: np.array
        Set of ids for objects available for querying.
    batch: int (optional)
        Number of objects to be chosen in each batch query.
        Default is 1.
    screen: bool (optional)
        If True display on screen the shift in index and
        the difference in estimated probabilities of being Ia
        caused by constraints on the sample available for querying.
    query_thre: float (optional)
        Maximum percentile where a spectra is considered worth it.
        If not queryable object is available before this threshold,
        return empty query. Default is 1.0.

    Returns
    -------
    query_indx: list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
    """
    if ensemble_probs.shape[0] != test_ids.shape[0]:
        raise ValueError('Number of probabiblities is different ' +
                         'from number of objects in the test sample!')

    # calculate distance to the decision boundary - only binary classification
    entropies, _ = compute_qbd_mi_entropy(ensemble_probs)

    # get indexes in decreasing order
    order = entropies.argsort()[::-1]

    # only allow objects in the query sample to be chosen
    flag = list(pd.Series(data=test_ids[order]).isin(queryable_ids))

    # check if there are queryable objects within threshold
    indx = int(len(flag) * query_thre)
    if sum(flag[:indx]) > 0:

        # arrange queryable elements in increasing order
        flag = np.array(flag)
        final_order = order[flag]

        if screen:
            print('*** Displacement caused by constraints on query****')
            print(' 0 -> ', list(order).index(final_order[0]))
            print(ensemble_probs[order[0]], '-- > ', ensemble_probs[final_order[0]])

        # return the index of the highest uncertain objects which are queryable
        return list(final_order)[:batch]

    else:
        return list([])


def main():
    return None

if __name__ == '__main__':
    main()
