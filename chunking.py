import nltk
nltk.download('brown')

# develop and import the following
import numpy as np
from nltk.corpus import brown

# split text into chunks
def splitter(content, num_of_words):
    words = content.split(' ')
    result = []
    
    # initialise the following programming lines to get the assigned variables
    current_count = 0
    current_words = []
    
    # start the iteration using words
    for word in words:
        current_words.append(word)
        current_count += 1
        
        # after getting the essential amount of words, reorganise the variables
        if current_count == num_of_words:
            result.append(' '.join(current_words))
            current_words = []
            current_count = 0
            
    # attach the chunks to the output variable:
    result.append(' '.join(current_words))
    return result

# import the data of Brown corpus and consider the first 10000 words:
if __name__ == '__main__':
    # read the data from the Brown corpus
    content = ' '.join(brown.words()[:10000])
    
    # describe the word size in every chunk
    # number of words in each chunk
    num_of_words = 1600
    
    # initiate a pair of significant variables
    chunks = []
    counter = 0
    
    # print the result by calling the splitter function
    num_text_chunks = splitter(content, num_of_words)
    print("Number of text chunks =", len(num_text_chunks))