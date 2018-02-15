#Lemmetizing is similar to stemming only the end word is more legible


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Normal plurals
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("Cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("eyes"))

print("\n\n")

# Forms of verbs
# By default, everything is set to pos="n".. remains same
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run", pos="v"))