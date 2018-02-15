#Named Entity Recognition is recognizing certain entities inside of an 
# document such as currency, time, date.
# Grouping Named entities.

# Source: https://pythonprogramming.net
'''
 Modifiers:

{1,3} = for digits, u expect 1-3 counts of digits, or "places"
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions
$ = matches at the end of string
^ = matches start of a string
| = matches either/or. Example x|y = will match either x or y
[] = range, or "variance"
{x} = expect to see this amount of the preceding code.
{x,y} = expect to see this x-y amounts of the precedng code

POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

Named Identity Examples:
ORGANIZATION	Georgia-Pacific Corp., WHO
PERSON 			Eddy, Obama, Gunther
LOCATION		New York, Washington
DATE			June, 2008-06-29
TIME			two fifty am, 1:20 p.m.
MONEY			175 million Canadian Dollers, GBP 10.40
PERCENT 		twenty pct, 18.75%
FACILITY 		Washington Monument, Stonehedge
GPE 			South East Asia, North East Asia

'''

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer 

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:

		for i in tokenized:
			words = nltk.word_tokenize(i)
			# After having split by word
			tagged = nltk.pos_tag(words)
			# We can tag each word'

			# namedEnt = nltk.ne_chunk(tagged)
			# Labels everything
			namedEnt = nltk.ne_chunk(tagged, binary=True)
			# Doesn't label, just defines the types

			namedEnt.draw()


	except Exception as e:
		print(str(e))

process_content()

