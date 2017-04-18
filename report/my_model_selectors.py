import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from numpy import asarray

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic_score = None
        min_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # Fit model using n compoments
                model = self.base_model(n)
                # Gets model score
                logL = model.score(self.X, self.lengths)

                d = model.n_features
                p = n ** 2 + 2 * n * d - 1

                # calculates BIC score
                bic_score = (-2 * logL + p * math.log(n))
                if min_bic_score is None or min_bic_score > bic_score:
                    min_bic_score = bic_score
                    min_model = model
            except:
                pass

        return min_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic_score = None
        max_model = None

        # Selects rest of the words (different than this word)
        rest_words = list(self.words)
        rest_words.remove(self.this_word)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # Fits model for this word
                model = self.base_model(n)
                # Gets score
                score = model.score(self.X, self.lengths)

                all_score = 0.0

                # For each word (not this word), fits a model and get score
                for w in rest_words:
                    # X, lengths corresponding to that word
                    X, lengths = self.hwords[w]

                    # Accumulates score
                    all_score = all_score+model.score(X, lengths)

                # Calculate DIC Score
                dic_score =  score - (all_score / (len(self.words) - 1))

                # Keep model with higher DIC score
                if max_dic_score is None or max_dic_score < dic_score:
                    max_dic_score = dic_score
                    max_model = model
            except:
                    pass

        return max_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = None
        max_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                all_score = 0.0
                qty = 0
                final_model = None

                if (len(self.sequences) >= 2):
                    # Generate K folds
                    folds = min(len(self.sequences),3)
                    split_method = KFold(shuffle=True, n_splits=folds)
                    parts = split_method.split(self.sequences)

                    for cv_train_idx, cv_test_idx in parts:
                        # Kfold information for train
                        X_train, lengths_train = asarray(combine_sequences(cv_train_idx, self.sequences))

                        # Kfold information for test
                        X_test, lengths_test = asarray(combine_sequences(cv_test_idx, self.sequences))

                        # Fit model with train data
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                        # Get score using test data
                        all_score = all_score+model.score(X_test,lengths_test)

                        qty = qty+1
                    # Calculate score
                    score = all_score / qty

                else:
                    # cant be fold
                    final_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    score = model.score(self.X, self.lengths)

                # Keep model with best score
                if max_score is None or max_score < score:
                    max_score = score
                    if final_model is None:
                        final_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                  random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    max_model = final_model

            except:
                pass

        return max_model
