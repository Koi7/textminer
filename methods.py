from dataprovider  import get_data
from dataprocessor import process_documents
from stopwords import list_stopwords
from operator  import itemgetter
import plsa
import gensim
import glob
import os

FILE_DEFAULT_GENSIM_WORD2ID_DICT  = 'tmp/gensim_word2id.dict'
FILE_DEFAULT_GENSIM_TF_IDF_MATRIX = 'tmp/gensim_tfidf.mm'
FILE_DEFAULT_GENSIM_LSA_RESULTS   = 'out/lsa_topic_modeling_results.txt'
FILE_DEFAULT_GENSIM_LDA_RESULTS   = 'out/lda_topic_modeling_results.txt'
FILE_DEFAULT_PLSA_RESULTS         = 'out/plsa_topic_modeling_results.txt'
PATH_TO_RAW_DATA                  = './data'
TOPICS_NUM                        = 10

def generate_temp_files_for_lda_or_lsa(documents_as_bag_of_words=[], word2id_file_path=FILE_DEFAULT_GENSIM_WORD2ID_DICT, tfidf_file_path=FILE_DEFAULT_GENSIM_TF_IDF_MATRIX):
	import gensim

	dictionary = gensim.corpora.Dictionary(documents_as_bag_of_words)
	dictionary.save_as_text(word2id_file_path)

	corpus = [dictionary.doc2bow(document) for document in documents_as_bag_of_words]
	gensim.corpora.MmCorpus.serialize(tfidf_file_path, corpus)

	return {
		'id2word': gensim.corpora.Dictionary.load_from_text(word2id_file_path),
		'corpus': gensim.corpora.MmCorpus(tfidf_file_path)
	}


def lsa(documents_as_bag_of_words=[], topics_num=TOPICS_NUM, write_results_to=FILE_DEFAULT_GENSIM_LSA_RESULTS):
	import gensim

	temp_files = generate_temp_files_for_lda_or_lsa(documents_as_bag_of_words)

	lsa_model = gensim.models.lsimodel.LsiModel(corpus=temp_files['corpus'], id2word=temp_files['id2word'], num_topics=TOPICS_NUM)

	topics = lsa_model.print_topics(TOPICS_NUM)

	with open(write_results_to, 'w', encoding='utf-8') as f:
		for topic in topics:
			f.write('{}: {}\n'.format(str(topic[0]), topic[1]))
		f.close()


def lda(documents_as_bag_of_words=[], topics_num=TOPICS_NUM, write_results_to=FILE_DEFAULT_GENSIM_LDA_RESULTS):
	import gensim

	temp_files = generate_temp_files_for_lda_or_lsa(documents_as_bag_of_words)

	lda_model = gensim.models.ldamodel.LdaModel(corpus=temp_files['corpus'], id2word=temp_files['id2word'], num_topics=TOPICS_NUM, update_every=1, chunksize=10000, passes=1)

	topics = lda_model.print_topics(TOPICS_NUM)

	with open(write_results_to, 'w', encoding='utf-8') as f:
		for topic in topics:
			f.write('{}: {}\n'.format(str(topic[0]), topic[1]))
		f.close()

def plsa(data_paths=[PATH_TO_RAW_DATA], topics_num=TOPICS_NUM, write_results_to=FILE_DEFAULT_PLSA_RESULTS):
	import plsa
	import glob
	import os
	# PLSA TOPIC MODELING
	corpus = plsa.Corpus() # instantiate corpus
	# iterate over the files in the directory.
	document_paths = data_paths
	for document_path in document_paths:
	    for document_file in glob.glob(os.path.join(document_path, '*.txt')):
	        document = plsa.Document(document_file) # instantiate document
	        document.split(list_stopwords(lang='en')) # tokenize
	        corpus.add_document(document) # push onto corpus documents list

	corpus.build_vocabulary()
	corpus.plsa(TOPICS_NUM, 1)

	V = len(corpus.vocabulary) 
	assert(TOPICS_NUM < V)
	f = open(write_results_to, "w")
	for k in range(TOPICS_NUM):
	    word_prob = corpus.topic_word_prob[k, :]
	    word_index_prob = []
	    for i in range(V):
	        word_index_prob.append([i, word_prob[i]])
	    word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) # sort by word count
	    f.write("Topic #" + str(k) + ":\n")
	    for i in range(TOPICS_NUM):
	        index = word_index_prob[i][0]
	        f.write(corpus.vocabulary[index] + " ")
	    f.write("\n")
	    
	f.close()




