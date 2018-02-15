import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

class VotedClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._calssifiers = classifiers
        # constructor function.
        # pass a list of classifiers 
        # _calssifiers is a list of classifiers egSVC, NuSVC etc.
    def classify(self, features):
        votes = []
        for c in self._calssifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        # We will call this and get who got the most votes
    def confidence(self, features):
        votes = []
        for c in self._calssifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        # counts how many occurances of that popular vote we got
        conf = choice_votes/len(votes)
        # how many of the chosen category over length of votes
        return conf

############
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()
############


############
word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()
############

def find_features(document):
    words = word_tokenize(document)
    # every single word will be included in this set of words
    features = {}
    #  empty dictionary
    for w in word_features:
        features[w] = (w in words)
        # if one these words is in the 3000 words
        #  it will give true, else ring false.
    return features


# ############
# featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
# featuresets = pickle.load(featuresets_f)
# featuresets_f.close()
# ############

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# Earlier it was just a words, with this, it will be a dictionary, with 
#  the words along with either it is positive or negative


random.shuffle(featuresets)
print(len(featuresets))
# to shuffle up, and not have bunch of consistant pos, and neg

# Build a training set

# Positive data example:
training_set = featuresets[10000:]
# 1900 feature sets to train
testing_set = featuresets[:10000]
# 1900 feature sets to test against

# (1) Origanal NLTK Naive Bayes
############
open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()
############

# (2) MNB Classifier
############
open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
############


# (3) Bernoulli Classifier
############
open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
############

#  (4) Logistic Regression Classifier
############
open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
############

#  (5) Linear SV Clasifier
############
open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
############

#  (6) Stochastic Gradient Descent Classifierc
############
open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()
############



voted_classifier = VotedClassifier(classifier, MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDC_classifier,LinearSVC_classifier)
# Find the most voted 
# Doesn't need pickling

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)