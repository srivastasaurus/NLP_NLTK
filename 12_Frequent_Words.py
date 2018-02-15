import nltk
import random
from nltk.corpus import movie_reviews

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

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# Earlier it was just a words, with this, it will be a dictionary, with 
#  the words along with either it is positive or negative



