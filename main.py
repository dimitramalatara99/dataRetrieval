import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


from Inverted_Index import InvertedIndex
from vsm2 import VectorSpaceModel
from Boolean_Model import BooleanModel
nltk.download('stopwords')

stemmer = PorterStemmer()
stopwords = stopwords.words("english")
iv = InvertedIndex()

def preprocess():

    doc_path = os.getcwd()
    docs_folder_path = os.path.join(doc_path, "collection", "docs")
    docs = []

    for doc_id in os.listdir(docs_folder_path):
        docs.append(doc_id)
        file_path = os.path.join(docs_folder_path, doc_id)
        with open(file_path, 'r', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                if word not in stopwords and word != "":
                    stem = stemmer.stem(word.lower())
                    iv.add_word(stem, doc_id)

    doc_count = len(docs)
    return set(map(int, docs)), doc_count

def preprocess_queries():

    doc_path = os.getcwd()
    queries_folder_path = os.path.join(doc_path, "collection", "Queries.txt")
    relevant_folder_path = os.path.join(doc_path, "collection", "Relevant.txt")

    query_data = {}
    query_id = 1

    with open(queries_folder_path, encoding='utf-8') as queries_file:
        for line in queries_file:
            tokens = line.strip().split()
            processed_tokens = [word.lower() for word in tokens if word.lower() not in stopwords]
            stemmed_tokens = [stemmer.stem(word) for word in processed_tokens]
            tf = Counter(stemmed_tokens)
            total_terms = len(processed_tokens)
            query_data[f"Q{query_id}"] = {
                "stemmed_tokens": stemmed_tokens,
                "tf": tf,
                "total_terms": total_terms
            }
            query_id += 1


    loaded_relevant_docs = []
    with open(relevant_folder_path, 'r', encoding='utf-8') as relevant_file:
        for line in relevant_file:
            relevant_ids = list(map(int, line.strip().split()))
            #exclude not existing ids
            filtered_relevant_ids = [doc_id for doc_id in relevant_ids if doc_id in doc_ids]
            loaded_relevant_docs.append(filtered_relevant_ids)

    return query_data, loaded_relevant_docs



def run_boolean():
    boolean_model = BooleanModel(iv.boolean_inv_index)
    boolean_results = []

    for query_id, query_data in queryTokens.items():
        tokens = query_data["stemmed_tokens"]
        boolean_query = " OR ".join(tokens)
        boolean_result = boolean_model.test_query(boolean_query)
        int_boolean_result = set(map(int, boolean_result))
        boolean_results.append(int_boolean_result)
    return boolean_results


def run_vsm(index, count, query_data):
    """
    Run VSM on all queries.
    :return: list of sets of retrieved document IDs
    """

    vsm = VectorSpaceModel(index, count)
    vsm.doc_tfidf()

    vsmResults = []
    for query_id, query_info in query_data.items():
        if not query_info:
            continue

        retrieved_list = vsm.search_tokens(query_info, top_k=100)
        ids = {int(doc_id_str) for doc_id_str, score in retrieved_list}
        vsmResults.append(ids)

    return vsmResults



# def run_vsm(index, docCount, queries_dict):
#
#     results = []
#    #print(f"Total docs: {doc_count}")
#     vsm = VectorSpaceModel(index, docCount)
#     vsm.doc_tfidf()
#
#     for q_id, token_list in queries_dict.items():
#         if not token_list:
#             # print(f"\n--- {q_id} is empty, skipping. ---")
#             continue
#
#         # print(f"\n--- {q_id}: '{token_list}' ---")
#         results = vsm.search_tokens(token_list, top_k=100)
#         # print("Results:", results)
#     return results

def precision_recall_calculation(results, relevant_docs,k):

    precisions = []
    recalls = []

    for retrieved, relevant in zip(results, relevant_docs):
        top_k_results = list(retrieved)[:k]
        relevant_count = len(set(top_k_results) & set(relevant))  # Intersection of retrieved and relevant
        precision = relevant_count / len(top_k_results) if len(top_k_results) > 0 else 0
        recall = relevant_count / len(relevant) if len(relevant) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    return recalls, precisions, avg_precision, avg_recall


def interpolate_pr_curve(recalls, precisions):
    """
    Interpolate precision-recall values at fixed recall points.
    """
    recall_points = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    interpolated_precisions = []

    for rp in recall_points:
        # Get the maximum precision for recall >= rp
        precision_at_rp = max((p for r, p in zip(recalls, precisions) if r >= rp), default=0)
        interpolated_precisions.append(precision_at_rp)

    return recall_points, interpolated_precisions



if __name__ == '__main__':
    # Step 1: Preprocess documents
    doc_ids, doc_count = preprocess()

    # Step 2: Preprocess queries and load relevant documents
    queryTokens, loadedRelevantDocs = preprocess_queries()

    # Step 3: Run Boolean Model
    booleanResults = run_boolean()

    # Step 4: Run VSM
    vsm_results = run_vsm(iv.vs_inv_index, doc_count, queryTokens)

    # Boolean Model Precision-Recall
    bool_recalls, bool_precisions, bool_avg_recall, bool_avg_precision = precision_recall_calculation(booleanResults,
                                                                                                      loadedRelevantDocs,100)

    # VSM Precision-Recall
    vsm_recalls, vsm_precisions, vsm_avg_recall, vsm_avg_precision = precision_recall_calculation(vsm_results,
                                                                           loadedRelevantDocs, 100)

    # Interpolation
    bool_interpolated_recall, bool_interpolated_precision = interpolate_pr_curve(bool_recalls, bool_precisions)
    vsm_interpolated_recall, vsm_interpolated_precision = interpolate_pr_curve(vsm_recalls, vsm_precisions)

    # Plot
    plt.figure(figsize=(10, 6))

    # Boolean Model PR Curve
    plt.plot(bool_recalls, bool_precisions, marker='o', linestyle='-', label='Boolean Model')

    # VSM PR Curve
    plt.plot(vsm_recalls, vsm_precisions, marker='s', linestyle='-', label='Vector Space Model')

    plt.title('Precision-Recall Comparison No Interpolation')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot
    plt.figure(figsize=(10, 6))

    # Boolean Model PR Curve
    plt.plot(bool_interpolated_recall, bool_interpolated_precision, marker='o', linestyle='-', label='Boolean Model')

    # VSM PR Curve
    plt.plot(vsm_interpolated_recall, vsm_interpolated_precision, marker='s', linestyle='-', label='Vector Space Model')

    plt.title('Precision-Recall Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()
