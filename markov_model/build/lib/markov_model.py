import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore")
words = ['female', 'male']

class markov_model(object):
    def __init__(self):
        def setup():
            def load_patterns(file):
                patterns = None
                sizes = np.zeros(len(words))
                counter = 0

                f = open(file, 'rb')
                data = f.readlines()

                stack = []
                for i in range(np.shape(data)[0]):
                    data2 = map(float, data[i].split())
                    data2 = np.reshape(data2, (1, -1))
                    if i == 0:
                        stack = data2
                    else:
                        stack = np.vstack((stack, data2))

                f.close()
                sizes[counter] = np.shape(stack)[0]
                counter += 1

                if patterns is None:
                    patterns = stack
                else:
                    patterns = np.vstack((patterns, stack))

                return patterns

            hidden = 1

            self.female_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(load_patterns('./csv_original_const/female.bin'))

            self.male_model = GaussianHMM(n_components=hidden, covariance_type="diag", n_iter=10000).fit(load_patterns('./csv_original_const/male.bin'))

        setup()
        self.number_of_components = 2

    def match(self, pattern):

        probabilities = np.zeros(2)
        probabilities[0] = self.female_model.score(np.reshape(pattern, (1, -1)))
        probabilities[1] = self.male_model.score(np.reshape(pattern, (1, -1)))

        probabilities = abs(probabilities)

        index, error = min(enumerate(probabilities), key=lambda x: x[1])

        if error < 9500:
            if index == 0:
                return 0
            elif index == 1:
                return 1
            else:
                return 4
        return -1