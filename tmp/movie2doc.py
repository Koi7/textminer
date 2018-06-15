import os

with open(os.path.join('../data/', 'movies.txt'), encoding="utf-8") as movies:
	contents = ''.join(movies.readlines())
	for index, movie in enumerate(contents.split('\n')):
		open(os.path.join('../data2/', 'document{}.txt'.format(index)), 'w', encoding='utf-8').write(movie)	
	