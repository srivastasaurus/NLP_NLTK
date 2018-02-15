# import nltk

# print(nltk.__file__)
# To print the location

# Find the data.py file.
# Search for nltk_data and open the corpus folder.
# the folder has various data sets that can be used.
# We will be using the movie_reviews data set & wordnet


from nltk.corpus import gutenberg
# Import the gutenberg Bible and do some operations on it.
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)
# Tokenize the entire bible

print(tok[5:15])
# Print from 5 through 15