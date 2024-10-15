# Copyright 2020 resspect software
# Author: The RESSPECT team
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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

__all__ = [
    'random_forest',
    'knn',
    'mlp',
    'svm',
    'nbg',
    'bootstrap_clf',
    'ResspectClassifer',
    'RandomForest',
    ]

class ResspectClassifer():
    """Base class that all built-in RESSPECT classifiers will inherit from."""

    def __init__(self, train_features, train_labels, test_features, **kwargs):
        """Base initializer for all RESSPECT classifiers.

        Parameters
        ----------
        train_features : array-like
            _description_
        train_labels : array-like
            _description_
        test_features : array-like
            _description_
        """
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.kwargs = kwargs

        #! Rename this after answering "Is shape[0] the number of objects or number of features/object?""
        self.num_test_data = self.test_features.shape[0]
        self._n_ensembles = 10
        self.n_labels = np.unique(self.train_labels).size
        self.ensemble_probs = np.zeros((self.num_test_data, self._n_ensembles, self.n_labels))

        self.classifier = None

    @property
    def n_ensembles(self):
        return self._n_ensembles

    @n_ensembles.setter
    def n_ensembles(self, value):
        self._n_ensembles = value
        self.ensemble_probs = np.zeros((self.num_test_data, self._n_ensembles, self.n_labels))

    def __call__(self):
        """Allows the user to call the class instance as a function.
        e.g. clf = SomeClassifier()
             predictions, _, _ = clf()
        """
        return self.predict(self.train_features, self.train_labels, self.test_features)

    def predict(self, train_features, train_labels, test_features):
        """Train and predict using the classifier.

        Parameters
        ----------
        train_features : array-like
            The features used for training, [n_samples, m_features].
        train_labels : array-like
            The training labels, [n_samples].
        test_features : array-like
            The features used for testing, [n_samples, m_features].

        Returns
        -------
        tuple(predictions, prob, classifier_instance)
            The classes and probabilities for the test sample.
        """
        self.classifier.fit(train_features, train_labels)
        predictions = self.classifier.predict(test_features)
        prob = self.classifier.predict_proba(test_features)

        return predictions, prob, self.classifier

    def bootstrap(self):
        """Convenience method that can be overridden by subclasses. Calls the
        bootstrap_ensemble method with the predict method as an argument.

        Returns
        -------
        tuple(predictions, prob, ensemble_probs, ensemble_clf)
            The classes and probabilities for the test sample.
        """
        return self.bootstrap_ensemble(self.predict)

    def bootstrap_ensemble(self, clf_function):
        """Create an ensemble of predictions by resampling the training data used
        to instantiate the classifier. Define the ensemble size by specifying the
        value for `n_ensembles`.

        e.g.:
        ```
        clf = SomeClassifier()
        clf.n_ensembles = 10
        clf.bootstrap_ensemble(clf.predict)
        ```

        Parameters
        ----------
        clf_function : Callable
            The function used to and predict with the classifier.

        Returns
        -------
        tuple(predictions, prob, ensemble_probs, ensemble_clf)
            The classes and probabilities for the test sample.
        """
        classifier_list = list()
        for i in range(self.n_ensembles):
            x_train, y_train = resample(self.train_features, self.train_labels)
            _, class_prob, clf = clf_function(x_train, y_train, self.test_features)

            classifier_list.append((str(i), clf))
            self.ensemble_probs[:, i, :] = class_prob

        ensemble_clf = PreFitVotingClassifier(classifier_list)
        class_prob = self.ensemble_probs.mean(axis=1)
        predictions = np.argmax(class_prob, axis=1)

        return predictions, class_prob, self.ensemble_probs, ensemble_clf


class RandomForest(ResspectClassifer):
    """RESSPECT-specific version of the sklearn RandomForestClassifier."""

    def __init__(self, train_features, train_labels, test_features, **kwargs):
        super().__init__(train_features, train_labels, test_features, **kwargs)

        self.n_estimators = kwargs.get('n_estimators', 100)
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, **self.kwargs)


def bootstrap_clf(clf_function, n_ensembles, train_features,
                  train_labels, test_features, **kwargs):
    """
    Train an ensemble of classifiers using bootstrap.

    Parameters
    ----------
    clf_function: function
        function to train classifier
    n_ensembles: int
        number of classifiers in the ensemble
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Prediction of the ensemble
    class_prob: np.array
        Average distribution of ensemble members
    ensemble_probs: np.array
        Probability output of each member of the ensemble
    """
    n_labels = np.unique(train_labels).size
    num_test_data = test_features.shape[0]
    ensemble_probs = np.zeros((num_test_data, n_ensembles, n_labels))
    classifier_list = list()
    for i in range(n_ensembles):
        x_train, y_train = resample(train_features, train_labels)
        predicted_class, class_prob, clf = clf_function(x_train,
                                                        y_train,
                                                        test_features,
                                                        **kwargs)
        #clf = clf_function(**kwargs)
        #clf.fit(x_train, y_train)
        #predicted_class = clf.predict(test_features)
        #class_prob = clf.predict_proba(test_features)
        
        classifier_list.append((str(i), clf))
        ensemble_probs[:, i, :] = class_prob

    ensemble_clf = PreFitVotingClassifier(classifier_list)
    class_prob = ensemble_probs.mean(axis=1)
    predictions = np.argmax(class_prob, axis=1)
    
    return predictions, class_prob, ensemble_probs, ensemble_clf


def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, n_estimators=100, **kwargs):
    """Random Forest classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Features from sample to be classified.
    n_estimators: int (optional)
        Number of trees in the forest. Default is 1000.
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
    clf = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob, clf

def knn(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):

    """K-Nearest Neighbour classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neighbors.KNeighborsClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = KNeighborsClassifier(**kwargs)

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob, clf


def mlp(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):

    """Multi Layer Perceptron classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neural_network.MLPClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf=MLPClassifier(**kwargs)

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob, clf

def svm(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):
    """Support Vector classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = SVC(probability=True, **kwargs)

    clf.fit(train_features, train_labels)          # train
    predictions = clf.predict(test_features)       # predict
    prob = clf.predict_proba(test_features)        # get probabilities

    return predictions, prob, clf
  

def nbg(train_features: np.array, train_labels: np.array,
                  test_features: np.array, **kwargs):

    """Naive Bayes classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf=GaussianNB(**kwargs)

    clf.fit(train_features, train_labels)         # fit
    predictions = clf.predict(test_features)      # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob, clf

class PreFitVotingClassifier(object):
    """Stripped-down version of VotingClassifier that uses prefit estimators"""
    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = [e[1] for e in estimators]
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions.astype('int'))
        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators')
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators')
        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T

def main():
    return None


if __name__ == '__main__':
    main()
