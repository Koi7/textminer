from dataprovider  import get_data
from dataprocessor import process_documents
from stopwords     import list_stopwords
from methods import lsa, lda, plsa


PATH_TO_DATASET = './dataset2'

print("Started...")

# get text from files in /data folder
documents = get_data(path=PATH_TO_DATASET);

# make bag-of-words
documents_as_bag_of_words = process_documents(documents, list_stopwords(lang='en'))

#run lsa
lsa(documents_as_bag_of_words)
# run lda
lda(documents_as_bag_of_words)
# run plsa
plsa([PATH_TO_DATASET])

print("Finished.")