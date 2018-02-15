# skikit learn has support vector machines, linear regression
# radial descent
# Theory and application of SVM
# https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/
# Tenserflow object Detection API playlist from sentdex


import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
# To import SciKitLEarn algorithms and functionalities in sklearn
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# We could Improve these algorithms by NOT using default parameters
# but we won't, we'll just use defaults

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC 

documents = [(list(movie_reviews.words(fileid)), category)
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)]
random.shuffle(documents) 

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# We need some sort of limit for words to train against


word_features = list(all_words.keys())[:3000]
# top 3000 words
# We can train against these top 3000 words 
# finding out which word is most common and pos
#  and which are common and neg

def find_features(document):
	words = set(document)
	# every single word will be included in this set of words
	features = {}
	#  empty dictionary
	for w in word_features:
		features[w] = (w in words)
		# if one these words is in the 3000 words
		#  it will give true, else ring false.
	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# Earlier it was just a words, with this, it will be a dictionary, with 
#  the words along with either it is positive or negative

# Build a training set

training_set = featuresets[:1900]
# 1900 feature sets to train
testing_set = featuresets[1900:]
# 1900 feature sets to test against


classifier_f = open("naivebayes.pickle", "rb")
# Open the saved pickle file with the training data. Read Bytes (rb)
classifier = pickle.load(classifier_f)
classifier_f.close()

# Using the static data set in classifier variable
print('\nOriginal Naive Bayes Algo accuracy (%): ', (nltk.classify.accuracy(classifier, training_set))*100)
# and We can find it's accuracy
classifier.show_most_informative_features(15)
#  Show 15 most informative features

# (1) MNB Classifier

MNB_Classifier = SklearnClassifier(MultinomialNB())
# We weill use the Multinomial Naive Bayes here
MNB_Classifier.train(training_set)
# This time we will make it learn using the sklearn classifier
print('\n MNB_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(MNB_Classifier, training_set))*100)
# Now we replace clasifier with the MNB_Classifier

# (2) Gaussian Classifier
# Caution: Gaussian Has some issues, so we'll skip it
# GaussianNB_Classifier = SklearnClassifier(GaussianNB())
# GaussianNB_Classifier.train(training_set)
# print("GaussianNB_Classifier Algo accuracy (%): ", (nltk.classify.accuracy(GaussianNB_Classifier, training_set))*100)

# (3) Bernoulli Classifier

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print('\n BernoulliNB_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(BernoulliNB_Classifier, training_set))*100)


# We're using more than one algorithms becasue these algorithms
# are unreliable in their results, and in the end, we can have a 
# vote by considering the mean of all algorithms used, creating 
# RELIABILITY in the result

#  (4) Logistic Regression Classifier

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print('\n LogisticRegression_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(LogisticRegression_Classifier, training_set))*100)

#  (5) Stochastic Gradient Descent Classifier

SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print('\n SGDClassifier_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(SGDClassifier_Classifier, training_set))*100)

#  (6) Support Vector Classifier

SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print('\n SVC_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(SVC_Classifier, training_set))*100)

#  (7) Linear SV Clasifier

LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print('\n LinearSVC_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(LinearSVC_Classifier, training_set))*100)

#  (8) Nu-Support Vector Classifier

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print('\n NuSVC_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(NuSVC_Classifier, training_set))*100)

# Template
# classifier = SklearnClassifier(classifier_function())
# classifier.train(training_set)
# print("Classifier_Name Algo accuracy (%): ", (nltk.classify.accuracy(classifier, training_set))*100)
# The function can have parameters for accuracy, find more on sklearn website