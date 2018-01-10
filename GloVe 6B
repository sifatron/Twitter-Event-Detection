# GlovE 6B Gensim VecSize300


import gensim
import glove

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


print('creating 300d glove')
glove_input_file = 'glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)



# load the Stanford GloVe model
filename = 'glove.6B.300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

# save and reload the model
root_path = r"C:/Users/SSK/Documents/MachineLearning/NLP/Socio-Sensing"
model.save(root_path + "GlovE 6B Gensim VecSize300")
# model = word2vec.KeyedVectors.load(root_path + "GlovE 6B Gensim VecSize300")


print('creating 100d glove')
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)



# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

# save and reload the model
root_path = r"C:/Users/SSK/Documents/MachineLearning/NLP/Socio-Sensing"
model.save(root_path + "GlovE 6B Gensim VecSize100")
# model = word2vec.KeyedVectors.load(root_path + "GlovE 6B Gensim VecSize100")



print('creating 50d glove')
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)



# load the Stanford GloVe model
filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

# save and reload the model
root_path = r"C:/Users/SSK/Documents/MachineLearning/NLP/Socio-Sensing"
model.save(root_path + "GlovE 6B Gensim VecSize50")
# model = word2vec.KeyedVectors.load(root_path + "GlovE 6B Gensim VecSize100")
