def get_data(path=''):
	import os
	import re
	documents = []
	for filename in os.listdir(path):
		with open(os.path.join(path, filename), encoding="utf-8") as f:
			documents.append(''.join(f.readlines()).replace('\n', ''))
	return documents