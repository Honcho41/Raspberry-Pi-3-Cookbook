import numpy as np
from nltk.corpus import brown
from chunking import splitter

import nltk
nltk.download('brown')

# define the main function and read the input data from Brown corpus
if __name__ == '__main__':
    content = ' '.join(brown.words()[:10000])
    
    # split the text into chunks
    num_of_words = 2000
    num_chunks = []
    count = 0
    texts_chunk = splitter(content, num_of_words)
    
    # build a vocabulary based on these text chunks
    for text in texts_chunk:
        num_chunk = {'index': count, 'text': text}
        num_chunks.append(num_chunk)
        count += 1
        
# extract a document matrix, which effectively counts the amount of incidences of each word in the document
from sklearn.feature_extraction.text import CountVectorizer

# extract the document term matrix
vectorizer = CountVectorizer(min_df = 0.5, max_df = 0.95)
matrix = vectorizer.fit_transform([num_chunk['text'] for num_chunk in num_chunks])

# extract the vocab and print it
vocabulary = np.array(vectorizer.get_feature_names())
print("Vocabulary:\n")
print(vocabulary)

# print the document term matrix
print("Document term matrix:")
chunks_name = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4']
formatted_row = '{:>12}' * (len(chunks_name) + 1)
print("\n", formatted_row.format('Word', * chunks_name), "\n")

# iterate throughout the words, and print the reappearance of every word in various chunks
for word, item in zip(vocabulary, matrix):
    # 'item' is a 'csr_matrix' data structure
    result = [str(x) for x in item.data]
    print(formatted_row.format(word, * result))