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
    query_tokens = {}
    query_id = 1
    doc_path = os.getcwd()
    docs_folder_path = os.path.join(doc_path, "collection", "docs")
    queries_folder_path = os.path.join(doc_path, "collection", "Queries.txt")

    for doc_id in os.listdir(docs_folder_path):
        file_path = os.path.join(docs_folder_path, doc_id)
        with open(file_path, 'r', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                if word not in stopwords:
                    stem = stemmer.stem(word)
                    iv.add_word(stem, doc_id)

    iv.document_count = len(os.listdir(docs_folder_path))
    iv.print()

    with open(queries_folder_path, encoding='utf-8') as queries_file:
        for line in queries_file:
            tokens = line.strip().split()
            processed_tokens = [word.lower() for word in tokens if word.lower() not in stopwords]
            stemmed_tokens = [stemmer.stem(word) for word in processed_tokens]
            query_tokens[f"Q{query_id}"] = stemmed_tokens
            query_id += 1
    return query_tokens


query_tokens = preprocess()

vsm = VectorSpaceModel(iv)

vsm.doc_tfidf()

vsm.query_tfidf( query_tokens)

results = vsm.search("info ret", top_k=1000)
#print(iv.document_count)
print("Top 5 results:", results)