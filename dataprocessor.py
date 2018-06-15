def process_documents(documents=[], stopwords=[]):
	""" Performs text processing for text mining. Makes bag-of-words from each document and returns list of bags-or-words. """

	# remove punctuation
	documents = remove_punctuation_and_digits(documents)
	# make bag-of-words
	documents_as_bag_of_words = make_bag_of_words(documents)
	# remove word that appear only once
	documents_as_bag_of_words = remove_uniq_words(documents_as_bag_of_words)
	# remove stop words
	documents_as_bag_of_words = remove_stopwords(documents_as_bag_of_words, stopwords)

	return documents_as_bag_of_words

def remove_punctuation_and_digits(documents=[]):
	""" Removes puncutation and digits from each document. """

	import re
	import string
	# regex to remove punctuation characters
	regex = re.compile('[%s]' % re.escape(string.punctuation + string.digits))
	documents_no_punctuation = []
	for document in documents:
		documents_no_punctuation.append(regex.sub('', document))

	return documents_no_punctuation

def make_bag_of_words(documents=[]):
	""" Splits documents' contents to separate words. """
	return [[word for word in document.lower().split()] for document in documents]

def remove_uniq_words(documents_as_bag_of_words=[]):
	from collections import defaultdict

	frequency = defaultdict(int)
	for document in documents_as_bag_of_words:
		for term in document:
			frequency[term] += 1

	return [[term for term in document if frequency[term] > 1] for document in documents_as_bag_of_words]

def remove_stopwords(documents_as_bag_of_words=[], stopwords=[]):
	""" Removes stopwords according to passed stopwords list. """
	return [[term for term in document if term not in stopwords] for document in documents_as_bag_of_words]
