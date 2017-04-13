import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                d = model.n_features
                p = n ** 2 + 2 * n * d - 1

                bic_score = (-2 * logL + p * math.log(n))
                if min_bic_score == None or min_bic_score > bic_score:
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

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)

                rest_words = list(self.words)
                rest_words.remove(self.this_word)

                all_score = 0.0

                for w in rest_words:
                    #model = self.base_model(n)
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                    all_score = all_score+model.score(self.hwords[w][0], self.hwords[w][1])

                dic_score =  score - all_score / len(rest_words)

                if max_dic_score == None or max_dic_score < dic_score:
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

        split_method = KFold(min(len(self.sequences), 3))
        max_score = None
        max_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                all_score = 0.0
                parts = split_method.split(self.sequences)
                qty = 0
                for cv_train_idx, cv_test_idx in parts:
                    X, lengths = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)

                    all_score = all_score+model.score(X_test,lengths_test)
                    qty = qty + 1

                score = all_score / qty

                if max_score == None or max_score < score:
                    max_score = score
                    max_model = model

            except:
                pass

        return max_model
