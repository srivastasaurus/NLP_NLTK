# words that have no interpretational meaniong, like a, an, the, is, 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing off stop word filteration."

stop_words = set(stopwords.words("english"))

# print(stop_words)

words = word_tokenize(example_sentence)

# filtered_sentence = []

# for w in words: 
# 	if w not in stop_words:
# 		filtered_sentence.append(w)

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)