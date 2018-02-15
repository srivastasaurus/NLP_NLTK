

# tokeninzing - grouoing things together... 
# word tokenizer, sentence tokenizer

# lexcon and corpora
# corpora - body of text, eg. medical journals, math journals. Similar type of text
# lexicon - dictionary. Words and their meanings. Invester speak vs regular speak
#         eg. "bullshit" has two meanings.


from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr.Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish blue. You should not eat cardboard."

# print(sent_tokenize(example_text))

for i in word_tokenize(example_text):
	print(i)
