import math

class VectorSpaceModel:
    def __init__(self, inverted_index, doc_count):

        self.inverted_index = inverted_index
        self.doc_count = doc_count
        self.documents = {}  # {doc_id: {term: tfidf_value}}


    def doc_tfidf(self):

        vs_index = self.inverted_index

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
                df = vs_index[term]["DF"]
                if df == 0 or self.doc_count == 0:
                    idf = 0
                else:
                    idf = math.log10(self.doc_count / df)

                self.documents[doc_id][term] = tf * idf

    def query_tfidf(self, query_info):

        stemmed_tokens = query_info["stemmed_tokens"]
        term_frequencies = query_info["tf"]
        total_terms = query_info["total_terms"]

        #  TF-IDF for each query term
        query_tfidf = {}
        for term in stemmed_tokens:
            tf = term_frequencies[term] / total_terms if total_terms > 0 else 0

            #  (IDF) query doesnt have DF so we use DF from the collection of docs
            df = self.inverted_index.get(term, {}).get("DF", 0)
            idf = math.log10(self.doc_count / df) if df > 0 else 0

            # TF-IDF
            query_tfidf[term] = tf * idf

        return query_tfidf

    def cosine_similarity(self, doc_vector, query_vector):

        # cosine similarity between q and d

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

    def search_tokens(self, query_data, top_k=10):

        # query vectors
        query_vec = self.query_tfidf(query_data)

        #  similarity for each doc
        scores = []
        for doc_id, doc_vector in self.documents.items():
            sim = self.cosine_similarity(doc_vector, query_vec)
            doc_id_stripped = doc_id.lstrip("0") or "0"
            #strips zeros(not the all 0 case)
            scores.append((doc_id_stripped, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and top_k < len(scores):
            scores = scores[:top_k]

        # results
        print("---Descending ranking by similarity---")
        for  doc_id_stripped, score in scores:
            print(f" {doc_id_stripped} | Score: {score:.4f}")

        return scores