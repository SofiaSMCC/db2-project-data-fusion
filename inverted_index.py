import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os
import json
import math

nltk.download('punkt')

class InvertedIndex:
    def __init__(self, dataset):
        self.stemmer = SnowballStemmer('english')
        self.stoplist = []

        with open("utils/stoplist.txt", encoding="latin1") as file:
            self.stoplist = set(line.rstrip().lower() for line in file)
        self.stoplist.update(['?', '-', '.', ':', ',', '!', ';'])

        self.data = pd.read_csv(dataset)
        self.dataset = { row['song_id']: self.pre_processing(row['lyrics']) for _, row in self.data.iterrows() }
        self.path = 'utils/inverted_index'
        self.total_docs = len(self.dataset)
        self.total_blocks = 0
        self.block_limit = 500

    def pre_processing(self, text, stemming=True):
        text = text.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        words = [word for word in words if word not in self.stoplist]
        if stemming:
            words = [self.stemmer.stem(word) for word in words]
        return words

    def spimi_invert(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        block_count = 0
        dictionary = defaultdict(dict)
        for i, (doc_id, words) in enumerate(self.dataset.items()):
            for word in words:
                if word not in dictionary:
                    dictionary[word] = { doc_id: 1 }
                else:
                    if doc_id in dictionary[word]:
                        dictionary[word][doc_id] += 1
                    else:
                        dictionary[word][doc_id] = 1

            if (i + 1) % self.block_limit == 0:
                self.save_temp_block(dictionary, block_count)
                dictionary.clear()
                block_count += 1

        self.total_blocks = block_count
        if dictionary:
            self.save_temp_block(dictionary, block_count)
        
        self.merge_all_blocks()

    def merge_blocks(self, start, finish):
        dictionary = defaultdict(dict)  

        for i in range(start, min(finish + 1, self.total_blocks)):
            with open(f"{self.path}/temp_block_{i}.json", "r") as file:
                json_data = json.load(file)
                for word, postings in json_data.items():
                    dictionary[word].update(postings)
        
        sorted_dict = sorted(dictionary.keys())
        
        total_elements = len(sorted_dict)
        num_blocks = finish - start + 1
        block_size, remainder = divmod(total_elements, num_blocks)
        
        temp_dict = {}
        current_block_size = block_size + (1 if remainder > 0 else 0)
        remainder -= 1
        block_count = 0
        
        for word in sorted_dict:
            temp_dict[word] = dictionary[word]
            
            if len(temp_dict) == current_block_size:
                self.save_block(temp_dict, start + block_count)
                temp_dict = {}
                block_count += 1
                
                if remainder > 0:
                    current_block_size = block_size + 1
                    remainder -= 1
                else:
                    current_block_size = block_size
        
        if temp_dict:
            self.save_block(temp_dict, start + block_count)

    def merge_all_blocks(self):        
        levels = math.ceil(math.log2(self.total_blocks))
        level = 1
        
        while level <= levels:
            step = 2**level
            for i in range(0, self.total_blocks, step):
                start = i
                finish = min(i + step - 1, self.total_blocks)
                self.merge_blocks(start, finish)
            level += 1

    def save_temp_block(self, dictionary, block_count):
        sorted_keys = sorted(dictionary.keys())
        sorted_values = { term: dictionary[term] for term in sorted_keys }
        with open(f"{self.path}/temp_block_{block_count}.json", "w") as file:
            json.dump(sorted_values, file, indent=4)
    
    def save_block(self, dictionary, block_count):
        sorted_keys = sorted(dictionary.keys())
        sorted_values = { term: dictionary[term] for term in sorted_keys }
        with open(f"{self.path}/block_{block_count}.json", "w") as file:
            json.dump(sorted_values, file, indent=4)

    def calculate_idf(self, all_docs_per_term):
        idf = {}
        for term, docs in all_docs_per_term.items():
            df = len(docs)
            idf[term] = math.log(self.total_docs / df)
        return idf

    """
    TF-IDF Calculation & Searching
    """

    def calculate_tfidf_for_document(self, data, idf):
        tfidf = {}
        for term, docs in data.items():
            for doc, tf in docs.items():
                tfidf_score = tf * idf.get(term, 0)
                if doc not in tfidf:
                    tfidf[doc] = {}
                tfidf[doc][term] = tfidf_score
        return tfidf

    def cosine_similarity(self, query, document):
        terms = set(query.keys()).union(document.keys())
        
        dot_product = sum(query.get(term, 0) * document.get(term, 0) for term in terms)
        query_magnitude = math.sqrt(sum(query.get(term, 0) ** 2 for term in terms))
        document_magnitude = math.sqrt(sum(document.get(term, 0) ** 2 for term in terms))
        
        if query_magnitude == 0 or document_magnitude == 0:
            return 0
        return dot_product / (query_magnitude * document_magnitude)

    def query_rank(self, query_terms, idf, tfidf):
        query_vector = {}
        for term in query_terms:
            if term in idf:
                query_vector[term] = idf[term]
        
        similarities = {}
        for doc, doc_tfidf in tfidf.items():
            similarity = self.cosine_similarity(query_vector, doc_tfidf)
            similarities[doc] = similarity
        
        ranked_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs
    
    def query_search(self, query, top_k):
        query_terms = self.pre_processing(query)

        all_docs_per_term = {}

        for file_name in os.listdir(self.path):
            if file_name.endswith('.json'):
                with open(os.path.join(self.path, file_name), "r") as file:
                    file_data = json.load(file)
                    for term, docs in file_data.items():
                        if term not in all_docs_per_term:
                            all_docs_per_term[term] = {}
                        for doc, tf in docs.items():
                            if doc not in all_docs_per_term[term]:
                                all_docs_per_term[term][doc] = 0
                            all_docs_per_term[term][doc] += tf

        idf = self.calculate_idf(all_docs_per_term)

        tfidf = {}
        for file_name in os.listdir(self.path):
            if file_name.endswith('.json'):
                with open(os.path.join(self.path, file_name), "r") as file:
                    file_data = json.load(file)
                    file_tfidf = self.calculate_tfidf_for_document(file_data, idf)
                    for doc, doc_tfidf in file_tfidf.items():
                        if doc not in tfidf:
                            tfidf[doc] = {}
                        for term, score in doc_tfidf.items():
                            tfidf[doc][term] = score

        ranked_docs = self.query_rank(query_terms, idf, tfidf)
        return ranked_docs[:top_k]

if __name__ == "__main__":
    index = InvertedIndex('utils/dataset.csv')

    index.spimi_invert()
    print(index.query_search('Goodbye yellow brick road', 5))