import re
import numpy as np
import tensorflow as tf
import keras
import nltk
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# for tokenization sentences
nltk.download('punkt')
nltk.download('stopwords')

"""
побудувати text classifier для задачі sentiment classification
датасет пропоную https://keras.io/api/datasets/imdb/
як модель -  використати google-news (з практики).
якщо з практики, пропоную наступний підхід:
1) розбити рев'ю на речення
2) обробити речення (викинути стоп-слова, і тд)
3) знайти ембедінги речень як середнє ембедінгів слів
4) побудувати ембедінги "позитивності" P і "негативності" N як середнє слів, 
які відповідають за позитив і негатив (напр. [good, nice, great, super, cool, etc] - positive)
5) для кожного рев'ю обчислити відстань від рев'ю до P, N. клас, до якого відстань мінімальна і буде відповіддю
модель тут
﻿https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
також вітаються якісь ідеї щодо покращення моделі. наприклад щодо побудови P, N.
"""

# download imdb dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=20000)

# download googleNews model
model_path = 'GoogleNews-vectors-negative300.bin.gz'
w2v = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=20000)

# preparing index for decoding back to text
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


def decode_review(encoded_review):
    """
funct for decoding the review text. "?" - word not fount in dictionary. "i-3" - first 3 index reserved
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# # decoding first review
# first_review_decoded = decode_review(x_train[0])
# print('first review (decoded):', first_review_decoded)
#
# # splitting review into sentences
# sentences = sent_tokenize(first_review_decoded)
# print('sentences in second review:', sentences)

stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(text):
    """
delete .,!? ect
casting to lowercase
delete stopwords
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop])
    return text


# cleaned_sentences = [clean_text(sentence) for sentence in sentences]
# print('cleaned Sentences:', cleaned_sentences)


def get_emb(text):
    """
creaing array for each word, check if list not empty, return mean of array
    """
    word_vectors = [w2v[i] for i in text.split() if i in w2v]
    if not word_vectors:
        return np.zeros(w2v.vector_size)
    return np.mean(np.array(word_vectors), axis=0)


# embeddings = [get_emb(i) for i in cleaned_sentences if i]
#
# # showing embeddings
# for i, embedding in enumerate(embeddings):
#     print(f"embedding for sentence {i + 1}: {embedding}")

pos_words = ['good', 'great', 'excellent', 'amazing', 'positive', 'happy', 'joy', 'wonderful', 'best', 'love']
neg_words = ['bad', 'worst', 'terrible', 'awful', 'negative', 'sad', 'anger', 'poor', 'worse', 'hate']

pos_emb = [get_emb(i) for i in pos_words if i]
neg_emb = [get_emb(i) for i in neg_words if i]


# for i, positive_vector in enumerate(positive_vector):
#     print(f"embedding for positive words {i + 1}: {positive_vector}")

def classify_review(rev_emb, pos_emb, neg_emb):
    """
measure distance between review and P,N
    """
    dist_pos = np.linalg.norm(rev_emb - pos_emb)
    dist_neg = np.linalg.norm(rev_emb - neg_emb)
    return 'positive' if dist_pos < dist_neg else 'negative'


results = []
correct_count = 0
for review, label in zip(x_train, y_train):
    decoded_review = decode_review(review)
    cleaned_review = clean_text(decoded_review)
    rev_emb = get_emb(cleaned_review)
    classification = classify_review(rev_emb, pos_emb, neg_emb)
    results.append((classification, 'positive' if label == 1 else 'negative'))
    correct_count += (classification == ('positive' if label == 1 else 'negative'))

print('first review(classified/real):', results[0])

accuracy = correct_count / len(x_train)
print(f"Accuracy: {accuracy} %")
