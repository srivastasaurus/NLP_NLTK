# We will now try to attempt creating our own text classifier
# for sentiment analysis - emotion out of piece of text
# positive or negative
# eg. spam or not span in inbox


import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)]
# A list of tuples i.e words i.e features and categories
#  In ML features make up the elements which are used to create
# categories, which are further used as training data 
# for category in movie_reviews.categories():
# 	for fileid in movie_reviews.fileids(category):
# 		documents.append(list(movie_reviews.words(fileid)))
# 		documents.append(category)

random.shuffle(documents) 

# documents is our training data

# Now, we will try to find the most popular words for each coment
#  and we will specify which ones come in popular and which one
#  come in negative. Then, the next time a review comes, we
#  simply search for those words, and that way we can classify it
#  as positive or negative

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print("15 Most common words")
print(all_words.most_common(15))
# Unfortunately nltk considers punctuiation and articles as words :/

print("\nNumber of times 'stupid' appears:")
print(all_words["stupid"])
# Displays number of times "stupid" appears