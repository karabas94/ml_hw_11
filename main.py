import tensorflow as tf
import keras
import nltk
import gensim
from gensim.models import KeyedVectors

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
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# print first review and labels
print('first review: ', x_train[0])
print('label of review: ', y_train[0])

# download googleNews model
model_path = 'D:\\GoogleNews-vectors-negative300.bin'
w2v = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=20000)
