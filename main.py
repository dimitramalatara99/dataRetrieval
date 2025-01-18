import os
from Inverted_Index import InvertedIndex
from vsm2 import VectorSpaceModel
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')


stemmer = PorterStemmer()
stopwords = stopwords.words("english")
iv = InvertedIndex()


doc_path = os.getcwd()
docs_folder_path = os.path.join(doc_path, "collection", "docs")


def preprocess():
    documents = []
    doc_path = os.getcwd()
    docs_folder_path = os.path.join(doc_path, "collection", "docs")
    iv.document_count = len(os.listdir(docs_folder_path))

    for doc_id in os.listdir(docs_folder_path):
        file_path = os.path.join(docs_folder_path, doc_id)
        with open(file_path, 'r', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                stem = stemmer.stem(word)
                if stem not in stopwords:
                    iv.add_word(stem, doc_id)

    # iv.print()


preprocess()

vsm = VectorSpaceModel(iv)
vsm.doc_tfidf()

results = vsm.search("", top_k=5)
#print(iv.document_count)
print("Top 5 results:", results)