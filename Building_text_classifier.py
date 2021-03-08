
# import training libraries and modules
from sklearn.datasets import fetch_20newsgroups

# dictionary for categories
category_mapping = {'misc.forsale': 'Sellings', 'rec.motorcycles': 'Motorbikes', 'rec.sport.baseball': 'Baseball', 'sci.space': 'OuterSpace'}

# allocate content to a training variable
training_content = fetch_20newsgroups(subset = 'train', categories = category_mapping.keys(), shuffle = True, random_state = 7)

# perform feature extraction to extract the main words in the text
from sklearn.feature_extraction.text import CountVectorizer

vectorizing = CountVectorizer()
train_counts = vectorizing.fit_transform(training_content.data)
print("Dimensions of training data:", train_counts.shape)

# train the classifier:
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

input_content = [
    "The curveballs or right handed pitchers tend to curve to the left",
    "Supernovae are just exploding stars",
    "This two-wheeler is really good on slippery roads",
    "I listed it on a well-known auction site for offers"
]

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

# implement the multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, training_content.target)
input_counts = vectorizing.transform(input_content)
input_tfidf = tfidf_transformer.transform(input_counts)

# predict the output categories
categories_prediction = classifier.predict(input_tfidf)

for sentence, category in zip(input_content, categories_prediction):
    print('\nInput:', sentence, '\nPredicted category:', category_mapping[training_content.target_names[category]])

