# 90% of the job in machine learning is just organization of data, cleaning up of data
# and analysis of the data is cherry on top, at the very end
#  nltk helps you do the structuring and organizing

# Stemming - Take words, and take the root stem of the word. 'Writing' -> 'write'
# Different variations of words: but the meaning is unchanged
# eg: I was riding in the car
#     I was taking a ride in the car.
# Algorithm used is called Porter stemmer

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

# for w in example_words:
# 	print(ps.stem(w))

new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners havee pythoned poorly at once."

words = word_tokenize(new_text)

for w in words:
	print(ps.stem(w))