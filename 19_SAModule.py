# Here we will attempt to tie this all up into a module,
# So that we can do a live Sentiment Analysis taking data from Twitter
# We'll just pickle and move on
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
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


short_pos = open("short_reviews/positive.txt").read()
short_neg = open("short_reviews/negative.txt").read()

all_words = []
documents = []

# j is adject, r is adverb, and v is verb
# allowed word types = ["J", "R", "V"]
allowed_word_types = ["J"]
# We're only allowing adjectives!!

for r in short_pos.split('\n'):
	# split by new line
	documents.append((r,"pos"))
	# Documents is a tuple, and it consists:
	# review, whether it is pos or neg
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for r in short_neg.split('\n'):
	documents.append((r,"neg"))
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

############
save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()
############

all_words = nltk.FreqDist(all_words)
# Then we find it's frequency distribution


# We need some sort of limit for words to train against
word_features = list(all_words.keys())[:5000]
# This time we take top 5000 words

############
save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
# Saving the words
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

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
# Using the static data set in classifier variable
print('\nOriginal Naive Bayes Algo accuracy (%): ', (nltk.classify.accuracy(classifier, training_set))*100)
# and We can find it's accuracy
classifier.show_most_informative_features(15)
#  Show 15 most informative features

############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
############

# (1) MNB Classifier

MNB_Classifier = SklearnClassifier(MultinomialNB())
# We weill use the Multinomial Naive Bayes here
MNB_Classifier.train(training_set)
# This time we will make it learn using the sklearn classifier
print('\n MNB_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(MNB_Classifier, training_set))*100)
# Now we replace clasifier with the MNB_Classifier
############
save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_Classifier, save_classifier)
save_classifier.close()
############
# (2) Gaussian Classifier

# Doesn't work

# (3) Bernoulli Classifier

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print('\n BernoulliNB_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(BernoulliNB_Classifier, training_set))*100)
############
save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_Classifier, save_classifier)
save_classifier.close()
############
#  (4) Logistic Regression Classifier

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print('\n LogisticRegression_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(LogisticRegression_Classifier, training_set))*100)
############
save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_Classifier, save_classifier)
save_classifier.close()
############
#  (5) Stochastic Gradient Descent Classifierc
SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print('\n SGDClassifier_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(SGDClassifier_Classifier, training_set))*100)
############
save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_Classifier, save_classifier)
save_classifier.close()
############
#  (6) Support Vector Classifier

# Bad score

#  (7) Linear SV Clasifier

LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print('\n LinearSVC_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(LinearSVC_Classifier, training_set))*100)
############
save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_Classifier, save_classifier)
save_classifier.close()
############
#  (8) Nu-Support Vector Classifier

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print('\n NuSVC_Classifier Algo accuracy (%): ', (nltk.classify.accuracy(NuSVC_Classifier, training_set))*100)

voted_classifier = VotedClassifier(classifier, MNB_Classifier,BernoulliNB_Classifier,LogisticRegression_Classifier,SGDClassifier_Classifier,LinearSVC_Classifier,NuSVC_Classifier)
# Find the most voted 
# Doesn't need pickling

def sentiment(text):
	feats = find_features(text)
	return voted_classifier(feats)
# quick functin that returns features of the text