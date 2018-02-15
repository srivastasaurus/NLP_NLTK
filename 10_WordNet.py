# Largest capability Corpora
#  Find synonyms, antonyms, and all other cool stuff with words

from nltk.corpus import wordnet

syns = wordnet.synsets("programs")

print("\n\n")

print("All:")
print(syns)
# Displaying sys sets
print("Name of FIrst Syn Set:")
print(syns[0].name())
# Displaying First 
print("First Element:")
print(syns[0].lemmas())
# Displaying First Element
print("For Plan:")
print(syns[0].lemmas()[0].name())

print("\n\n")

print("Name of syn set, definition, and examples:")
print(syns[0].name())
print(syns[0].definition())
print(syns[0].examples())

print("\n\n")

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())

print("Synonyms: ")
print(set(synonyms))
print("Antonyms: ")
print(set(antonyms))

print("\n\n")

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

# We can compare the similarity between the two words
print("Displaying Similarity of 'ship' and 'boat':")
print(w1.wup_similarity(w2))
# Wu and Palmer wrote a paper on similarities between words
#  Comparing similarity of w1 to w2

w3 = wordnet.synset("cat.n.01")
print("Displaying Similarity of 'ship' and 'cat':")
print(w1.wup_similarity(w3))

w4 = wordnet.synset("car.n.01")
print("Displaying Similarity of 'ship' and 'car':")
print(w1.wup_similarity(w4))

w5 = wordnet.synset("cactus.n.01")
print("Displaying Similarity of 'ship' and 'cactus':")
print(w1.wup_similarity(w5))

# If you have to figure out whether two documents are same or not,
# or to differenciate two documents, this can be used

# News bots that write articles automatically
# Google can, however do the opposite and check.