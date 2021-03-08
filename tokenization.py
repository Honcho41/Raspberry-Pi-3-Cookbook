# import and download modules
import nltk
nltk.download('punkt')

# introduce sentence tokenisation:
from nltk.tokenize import sent_tokenize

text = "Tokenization is the process of dividing text into a set of meaningful pieces. These pieces are called tokens"

# form a new text tokeniser
tokenize_list_sent = sent_tokenize(text)
print("\nSentence tokenizer:")
print(tokenize_list_sent)

# form a new word tokenizer:
from nltk.tokenize import word_tokenize

print("\nWord tokenizer:")
print(word_tokenize(text))

# introduce a new WordPunct tokenizer:
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()
print("\nWord punct tokenizer:")
print(word_punct_tokenizer.tokenize(text))