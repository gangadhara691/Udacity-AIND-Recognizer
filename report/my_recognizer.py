import warnings
from asl_data import SinglesData


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
    # X, lengths = test_set.get_all_Xlengths()
    #
    # for word_id in range(0, len(test_set.get_all_Xlengths())):
    #
    #     current_sequence = test_set.get_item_sequences(word_id)
    #     current_length = test_set.get_item_Xlengths(word_id)
    #
    #     prob_word = dict()

    for test in range(0,len(test_set.get_all_Xlengths())):
        X, lengths = test_set.get_all_Xlengths()[test]

        max_score = None
        max_word = None
        prob_dict = dict()

        for word,model in models.items():
            #print(word,model)
            try:
                score = model.score(X, lengths)

            except:
                score = float("-Inf")

            if max_score == None or score > max_score:
                max_score = score
                max_word = word

            prob_dict[word] = score
        probabilities.append(prob_dict)
        guesses.append(max_word)

            # model = models[word]
        #
        #
        # for model_word in models[word]:
        #     score = model.score(current_sequence, current_length)
        #

    return (probabilities , guesses)
