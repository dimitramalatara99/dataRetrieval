import math

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class VectorSpaceModel:
    def __init__(self, inverted_index):

        self.inverted_index = inverted_index
        self.document_count = inverted_index.document_count
        self.documents = {}  # {doc_id: {term: tfidf_value}}

        #  query preprocess
        self.stopwords = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def doc_tfidf(self):

        vs_index = self.inverted_index.vs_inv_index

        doc_term_freqs = {}

        for term, info in vs_index.items():
            for doc_id, freq in info["TF"].items():
                if doc_id not in doc_term_freqs:
                    doc_term_freqs[doc_id] = {}
                doc_term_freqs[doc_id][term] = freq

        #  TF–IDF for each term in every doc
        for doc_id, freqs in doc_term_freqs.items():
            max_freq = max(freqs.values())  # highest r(aw) frequency in this doc
            self.documents[doc_id] = {}

            for term, freq in freqs.items():
                # tf=(freq / max freq)
                tf = freq / max_freq

                # idf=log(total_docs / DF)
                df = vs_index[term]["DF"]  # posa docs exoun to term
                ratio = self.document_count / df if df != 0 else None

                print(f"Term={term}, doc_count={self.document_count}, df={df}, ratio={ratio}")

                if df == 0 or self.document_count == 0:
                    idf = 0
                else:
                    idf = math.log10(self.document_count / df)

                self.documents[doc_id][term] = tf * idf

    def _preprocess_query(self, query_str):

        tokens = query_str.split()
        processed = []

        for token in tokens:
            token = token.lower().strip()
            if token in self.stopwords or token == "":
                continue
            stemmed = self.stemmer.stem(token)
            processed.append(stemmed)

        return processed

    def query_tfidf(self, query_tokens):

        if not query_tokens:
            return {}

        # raw frequencies in query
        freq_map = {}
        for t in query_tokens:
            freq_map[t] = freq_map.get(t, 0) + 1

        max_freq = max(freq_map.values())
        query_vec = {}

        for term, freq in freq_map.items():
            # TF for query
            tf = freq / max_freq

            # IDF from the index
            if term in self.inverted_index.vs_inv_index:
                df = self.inverted_index.vs_inv_index[term]["DF"]
                if df == 0 or self.document_count == 0:
                    idf = 0
                else:
                    idf = math.log10(self.document_count / df)
            else:
                # term not in collection then IDF=0
                idf = 0

            query_vec[term] = tf * idf

        return query_vec

    def cosine_similarity(self, doc_vector, query_vector):

        # cosine similarity between vectors

        numerator = 0.0
        for term, wq in query_vector.items():
            wd = doc_vector.get(term, 0.0)
            numerator += wd * wq

        # norms
        doc_norm = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
        query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))

        if doc_norm == 0 or query_norm == 0:
            return 0.0

        return numerator / (doc_norm * query_norm)

    def search(self, query_str, top_k=10):


        query_tokens = self._preprocess_query(query_str)

        # query vectors
        query_vec = self.query_tfidf(query_tokens)

        #  similarity for each doc
        scores = []
        for doc_id, doc_vector in self.documents.items():
            sim = self.cosine_similarity(doc_vector, query_vec)
            scores.append((doc_id, sim))

        # Sort descending by sim
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
