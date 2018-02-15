# How to save our trained alogithm, so that everytime we 
# we don't have to retrain it.
# Longest part of this algo is just loading the documents 

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
# Pickle is used to save documents

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
testing_sets = featuresets[1900:]
# 1900 feature sets to test against


'''
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# # we've trained it
# save_classifier = open("naivebayes.pickle", "wb")
# # Creating file name, and saving in bytes (wb - write bytes)
# pickle.dump(classifier, save_classifier)
# # dump all the data in theres
# save_classifier.close()
# # We save once and then comment out this section
# # and load the file the next time we run
# # We got 88% accuracy
'''


classifier_f = open("naivebayes.pickle", "rb")
# Open the saved pickle file with the training data. Read Bytes (rb)
classifier = pickle.load(classifier_f)
classifier_f.close()

# Using the static data set in classifier variable
print("Naive Bayes Algo accuracy (%): ", (nltk.classify.accuracy(classifier, training_set))*100)
# and We can find it's accuracy
classifier.show_most_informative_features(15)
#  Show 15 most informative features


# We could literally save the entire document using pickle and literally for anything else.