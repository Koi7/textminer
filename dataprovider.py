def get_data(path=''):
	import os
	import re
	import string
	documents = []
	for filename in os.listdir(path):
		with open(os.path.join(path, filename), encoding="utf-8") as f:
			file_contents = ''.join(f.readlines()).replace('\n', '')
			documents.append(file_contents)
	return documents