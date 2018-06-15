from dataprovider  import get_data
from dataprocessor import process_documents
from stopwords import list_stopwords
from operator  import itemgetter
import plsa
import gensim
import glob
import os


def print_topic_word_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print topic-word distribution to file and list @topk most probable words for each topic
    """
    print("Writing topic-word distribution to file: " + filepath)
    V = len(corpus.vocabulary) # size of vocabulary
    assert(topk < V)
    f = open(filepath, "w")
    for k in range(number_of_topics):
        word_prob = corpus.topic_word_prob[k, :]
        word_index_prob = []
        for i in range(V):
            word_index_prob.append([i, word_prob[i]])
        word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) # sort by word count
        f.write("Topic #" + str(k) + ":\n")
        for i in range(topk):
            index = word_index_prob[i][0]
            f.write(corpus.vocabulary[index] + " ")
        f.write("\n")
        
    f.close()
    
def print_document_topic_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print document-topic distribution to file and list @topk most probable topics for each document
    """
    print("Writing document-topic distribution to file: " + filepath)
    assert(topk <= number_of_topics)
    f = open(filepath, "w")
    D = len(corpus.documents) # number of documents
    for d in range(D):
        topic_prob = corpus.document_topic_prob[d, :]
        topic_index_prob = []
        for i in range(number_of_topics):
            topic_index_prob.append([i, topic_prob[i]])
        topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True)
        f.write("Document #" + str(d) + ":\n")
        for i in range(topk):
            index = topic_index_prob[i][0]
            f.write("topic" + str(index) + " ")
        f.write("\n")
        
    f.close()

# constants
TOPICS_AMOUNT = 10

# get text from files in /data folder
documents = get_data(path='data');

# make bag-of-words
documents_as_bag_of_words = process_documents(documents, list_stopwords(lang='en'))


dictionary = gensim.corpora.Dictionary(documents_as_bag_of_words)
dictionary_file = 'tmp/word2id.dict'
dictionary.save_as_text(dictionary_file)

corpus = [dictionary.doc2bow(document) for document in documents_as_bag_of_words]
tfidf_file = 'tmp/tfidf.mm'

gensim.corpora.MmCorpus.serialize(tfidf_file, corpus)
id2word = gensim.corpora.Dictionary.load_from_text(dictionary_file)
mm = gensim.corpora.MmCorpus(tfidf_file)

lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=TOPICS_AMOUNT)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=TOPICS_AMOUNT, update_every=1, chunksize=10000, passes=1)
# PLSA TOPIC MODELING
corpus = plsa.Corpus() # instantiate corpus
# iterate over the files in the directory.
document_paths = ['./data']
for document_path in document_paths:
    for document_file in glob.glob(os.path.join(document_path, '*.txt')):
        document = plsa.Document(document_file) # instantiate document
        document.split(list_stopwords(lang='en')) # tokenize
        corpus.add_document(document) # push onto corpus documents list

corpus.build_vocabulary()
print ("Vocabulary size:" + str(len(corpus.vocabulary)))
print ("Number of documents:" + str(len(corpus.documents)))

corpus.plsa(TOPICS_AMOUNT, 1)

#print corpus.document_topic_prob
#print corpus.topic_word_prob
#cPickle.dump(corpus, open('./models/corpus.pickle', 'w'))

print_topic_word_distribution(corpus, TOPICS_AMOUNT, TOPICS_AMOUNT, "./topic-word.txt")
print_document_topic_distribution(corpus, TOPICS_AMOUNT, TOPICS_AMOUNT, "./document-topic.txt")

lsi_topics = lsi.print_topics(TOPICS_AMOUNT)
lda_topics = lda.print_topics(TOPICS_AMOUNT)
