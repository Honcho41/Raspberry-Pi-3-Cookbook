# initialize the stemming process by importing modules
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# let's describe some words to consider
words = ['ability', 'baby', 'college', 'playing', 'is', 'dream', 'election', 'beaches', 'image', 'group', 'happy']

# identify a group of stemmers to be used
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']

# initialise the necessary tasks for the chosen stemmers
stem_porter = PorterStemmer()
stem_lancaster = LancasterStemmer()
stem_snowball = SnowballStemmer('english')

# format the table to print the results
formatted_row = '{:>16}' * (len(stemmers) + 1)
print('\n', formatted_row.format('WORD', *stemmers), '\n')

# repeatadly check the list of words and arrange them using chosen stemmers
for word in words:
    stem_words = [stem_porter.stem(word),
                  stem_lancaster.stem(word),
                  stem_snowball.stem(word)
                  ]
    print(formatted_row.format(word, *stem_words))
