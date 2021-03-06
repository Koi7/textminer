def list_stopwords(lang='en'):
	""" Makes list of stopwords from source file. """

	import os

	filename = 'stopwords_{}.txt'.format(lang)
	path = 'stopwords/'
	stopwords = []

	with open(os.path.join(path, filename), encoding="utf-8") as f:
		for word in f.readlines():
			stopwords.append(word.replace('\n', ''))

	return stopwords