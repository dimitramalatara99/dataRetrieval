class BooleanModel:
    def __init__(self, boolean_inv_index):
        self.boolean_inv_index = boolean_inv_index

    def get_documents(self, term):
        return self.boolean_inv_index.get(term, set())

    def handle_not(self, term_set):
        all_docs = self.get_all_documents()
        return all_docs - term_set

    def get_all_documents(self):
        all_docs = set()
        for docs in self.boolean_inv_index.values():
            all_docs.update(docs)
        return all_docs

    def process_query(self, query):
        tokens1 = query.lower().split()
        test = ""

        for token in tokens1:
            if token == "and":
                test += " & "
            elif token == "or":
                test += " | "
            elif token == "not":
                all_docs = set(doc_id for docs in self.boolean_inv_index.values() for doc_id in docs)
                test += f"(all_docs - "
            else:
                matching_docs = self.boolean_inv_index.get(token, set())
                test += f"{matching_docs}"

        test = test.replace("all_docs - ", "(all_docs - ") + ")" if "all_docs - " in test else test
        result_docs = eval(test)

        return result_docs