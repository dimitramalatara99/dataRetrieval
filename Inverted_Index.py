class InvertedIndex:
    def __init__(self):
        self.vs_inv_index = {}
        self.boolean_inv_index = {}
        self.document_count = 0

    def add_word_vs(self, word, doc_id):
        if word not in self.vs_inv_index:
            # Initialize word entry with document occurrence
            self.vs_inv_index[word] = {"DF": 1, "TF": {doc_id: 1}}
        else:
            if doc_id not in self.vs_inv_index[word]["TF"]:
                self.vs_inv_index[word]["DF"] += 1
                self.vs_inv_index[word]["TF"][doc_id] = 1
            else:
                self.vs_inv_index[word]["TF"][doc_id] += 1

    def add_word_boolean(self, word, doc_id):
        if word not in self.boolean_inv_index:
            self.boolean_inv_index[word] = {doc_id}
        else:
                self.boolean_inv_index[word].add(doc_id)

    def add_word(self, word, doc_id):
        self.add_word_vs(word, doc_id)
        self.add_word_boolean(word, doc_id)

    def print(self):
        print("vs_inv_index: ", self.vs_inv_index)
        print("DOC_count: ",self.document_count)
        print("boolean_inv_index: ", self.boolean_inv_index)