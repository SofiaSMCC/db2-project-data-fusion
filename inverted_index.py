import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import regex as re
import os
import json
import math

nltk.download('punkt')

class InvertedIndex:
    def __init__(self, dataset):
        self.stemmer = SnowballStemmer('english')
        self.stoplist = []

        with open("utils/stoplist.txt", encoding="utf-8") as file:
            self.stoplist = set(line.rstrip().lower() for line in file)
        self.stoplist.update(['?', '-', '.', ':', ',', '!', ';', '_'])

        self.data = pd.read_csv(dataset)
        self.dataset = { row['song_id']: self.pre_processing(row['lyrics']) for _, row in self.data.iterrows() }
        self.path = 'utils/inverted_index'
        self.total_docs = len(self.data)
        self.total_blocks = 0
        self.block_limit = 500

    def pre_processing(self, text, stemming=False):
        text = text.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        words = [
            word for word in words 
            if word.isascii() 
            and not re.search(r'\d', word)
            and '_' not in word 
            and word not in self.stoplist
        ]
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

    def search_in_blocks(self, word):
        self.total_blocks = len([name for name in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, name)) 
                                 and name.startswith('block_')])

        left, right = 0, self.total_blocks - 1
        while left <= right:
            mid = (left + right) // 2

            with open(f"{self.path}/block_{mid}.json", "r") as file:
                data = json.load(file)

                keys = list(data.keys())
                if word in data:
                    return data[word]
                elif word < keys[0]:
                    right = mid - 1
                elif word > keys[-1]:
                    left = mid + 1
                else:
                    break

        return -1
    
    def query_search(self, query, top_k=5):
        query = self.pre_processing(query)
        query_tf = { term: query.count(term) for term in query }
        query_tfidf = {}
        document_magnitude = {}
        scores = {}

        for term in query_tf:
            term_data = self.search_in_blocks(term)
            if term_data == -1:
                continue

            idf = math.log10(self.total_docs / len(term_data))
            query_tfidf[term] = math.log10(1 + query_tf[term]) * idf

            for doc_id, freq in term_data.items():
                doc_tfidf = math.log10(1 + freq) * idf
                scores[doc_id] = scores.get(doc_id, 0) + doc_tfidf * query_tfidf[term]
                document_magnitude[doc_id] = document_magnitude.get(doc_id, 0) + doc_tfidf ** 2

        query_magnitude = math.sqrt(sum(val ** 2 for val in query_tfidf.values()))
        for doc_id in scores:
            doc_vector_magnitude = math.sqrt(document_magnitude[doc_id])
            scores[doc_id] = scores[doc_id] / (query_magnitude * doc_vector_magnitude)

        return list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:top_k]

if __name__ == "__main__":
    index = InvertedIndex('utils/dataset.csv')

    # index.spimi_invert()
    # print(index.calc_tf('goodbye', 'spotify:track:3GzbUESYBLZWpEKuxXI5nV'))
    # print(index.query_search('Goodbye yellow brick road', 5))
    print(index.query_search('Goodbye yellow brick road', 5))