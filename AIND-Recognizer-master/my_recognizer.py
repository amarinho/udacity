import warnings
from asl_data import SinglesData


def __run_models(models: dict, x, lengths):
    """

    :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param x: array-like, shape (n_samples, n_features). Feature matrix of individual samples.
    :param lengths:  array-like of integers, shape (n_sequences, )
    :return: dict of trained models where each key is a word and value is Log Liklihood,
            and best guess words from the input dict
    """

    highest_score = float("-inf")
    log_l_dict = {}
    best_guess_word = None

    for word, hmm_model in models.items():

        try:
            score = hmm_model.score(x, lengths)
            if score > highest_score:
                log_l_dict[word] = score
                highest_score, best_guess_word = score, word
        except:
            log_l_dict[word] = None

    return log_l_dict, best_guess_word


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # implement the recognizer

    # The `build_test` method in `ASLdb` is similar to the `build_training` method
    # already presented, but there are a few differences:
    # - the object is type `SinglesData`
    # - the internal dictionary keys are the index of the test word rather than the word itself
    # - the getter methods are `get_all_sequences`, `get_all_Xlengths`, `get_item_sequences`
    #   and `get_item_Xlengths`

    sequences = test_set.get_all_sequences()

    for item in sequences:

        x, lengths = test_set.get_item_Xlengths(item)

        log_l_dict, best_guess_word = __run_models(models, x, lengths)

        probabilities.append(log_l_dict)
        guesses.append(best_guess_word)

    return probabilities, guesses
