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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    @staticmethod
    def __calculate_bic(log_l: float, num_of_parameters: int, num_of_data_points: int) -> float:
        """
        Calculates the BIC score:

          BIC = -2 * logL + p * logN
          Where:
            L is the likelihood of the fitted model, p is the number of parameters,
            and N is the number of data points

        see http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf

        :param num_of_parameters: number of parameters
        :param num_of_data_points: number of data points
        :return: BIC score
        """
        return (-2 * log_l) + num_of_parameters * np.log(num_of_data_points)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores

        # The hmmlearn library may not be able to train or score all models.
        # Implement try/except contructs as necessary to eliminate non-viable models.
        best_model = self.base_model(self.n_constant)

        lowest_bic = float("inf")

        # for n between self.min_n_components and self.max_n_components
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                # build a HMM instance
                # remodel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                #                      random_state=self.random_state, verbose=self.verbose)
                #
                # train an HMM by calling the fit method
                # The input is a matrix of concatenated sequences of observations (aka samples)
                # along with the lengths of the sequences
                # remodel.fit(self.X, self.lengths)
                remodel = self.base_model(n)

                # Compute the log probability
                log_l = remodel.score(self.X, self.lengths)

                # Calculate BIC score
                bic_score = self.__calculate_bic(log_l, n, len(self.X))

                # choose the best model
                if lowest_bic > bic_score:
                    lowest_bic, best_model = bic_score, remodel
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    @staticmethod
    def __calculate_dic(log_p_xi: float, m: int, sum_log_p_xi_excl_i: float) -> float:
        """
        Calculates the DIC score:

          DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

        see https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
        """
        return log_p_xi - ( 1/ (m - 1)) * sum_log_p_xi_excl_i

    def __calculate_sum_log_p_xi_excl_i(self, num_of_parameters) -> float:
        """
        Calculates sum of all log probabilities but except the prob. for this word
        :param num_of_parameters:
        :return: sum of all log probabilities but except the prob. for this word
        """

        words = list(self.words.keys())
        words.remove(self.this_word)

        sum_log_p_xi_excl_i = 0.
        skips = 0

        for word in words:
            try:
                model_selector = ModelSelector(self.words, self.hwords, word, self.n_constant,
                                               self.min_n_components, self.max_n_components,
                                               self.random_state, self.verbose)

                hmm_model = model_selector.base_model(num_of_parameters)

                sum_log_p_xi_excl_i += hmm_model.score(model_selector.X, model_selector.lengths)
            except:
                skips += skips

        return sum_log_p_xi_excl_i, skips

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # The hmmlearn library may not be able to train or score all models.
        # Implement try/except contructs as necessary to eliminate non-viable models.
        best_model = self.base_model(self.n_constant)

        highest_dic = float("-inf")
        number_of_words = len(self.words.keys())

        # for n between self.min_n_components and self.max_n_components
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                # build a HMM instance
                # remodel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                #                      random_state=self.random_state, verbose=self.verbose)
                #
                # train an HMM by calling the fit method
                # The input is a matrix of concatenated sequences of observations (aka samples)
                # along with the lengths of the sequences
                # remodel.fit(self.X, self.lengths)
                remodel = self.base_model(n)

                # Compute the log probability
                log_l = remodel.score(self.X, self.lengths)

                sum_log_p_xi_excl_i, skips = self.__calculate_sum_log_p_xi_excl_i(n)

                # Calculate DIC score
                dic_score = self.__calculate_dic(log_l, number_of_words - skips,
                                                 sum_log_p_xi_excl_i)

                # choose the best model
                if dic_score > highest_dic:
                    highest_dic, best_model = dic_score, remodel
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV

        # One technique for cross-validation is to break the training set into "folds" and rotate
        # which fold is left out of training. The "left out" fold scored.
        # This gives us a proxy method of finding the best model to use on "unseen data"

        # The hmmlearn library may not be able to train or score all models.
        # Implement try/except contructs as necessary to eliminate non-viable models.
        best_num_components = self.n_constant

        num_splits = 3
        if len(self.sequences) < num_splits:
            return self.base_model(best_num_components)

        split_method = KFold(n_splits=num_splits)
        best_score = float('-inf')

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                sum_log_l = 0.
                count_log_l = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))

                    # In order to run hmmlearn training using the X,lengths tuples on the
                    # new folds, subsets must be combined based on the indices given for the folds.
                    # A helper utility has been provided in the asl_utils module named
                    # combine_sequences for this purpose
                    x_train, len_x_train = combine_sequences(cv_train_idx, self.sequences)

                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=self.verbose)
                    hmm_model.fit(x_train, len_x_train)

                    x_test, len_x_test = combine_sequences(cv_test_idx, self.sequences)

                    test_score = hmm_model.score(x_test, len_x_test)

                    sum_log_l += test_score
                    count_log_l += 1

                if count_log_l > 0:
                    average_test_score = sum_log_l / count_log_l
                    if average_test_score > best_score:
                        best_score, best_num_components = average_test_score, n
            except:
                pass

        return self.base_model(best_num_components)
