'''
This is based on cnn35_64. This is after the first pilot. 
Changes:
-don't filter out # in the tokenizer, tokenize both together. or save tokenizer https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
-use 'number' w2v as representation for any digit
-shuffling problem should be check before advancing: plot random selection of conv1 layers. theys should all be 14 or 15.
-tune hyperparameters. 
'''
import datetime
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout
from keras.models import Model
import os
import data_helpers
import config
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.utils import np_utils
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)

# Parameters
# =====================================================================
categories = config.categories
verbose = config.verbose
toy = config.toy
save_checkpoints = config.save_checkpoints
plot_RSA = config.plot_RSA

if toy:
    # categories = categories[:4]
    epochs = 1 #it will probably need more.
else:
    epochs = config.epochs  # it will probably need more.

if config.local_or_cluster:
    categories = categories[:3]
    epochs=1
    verbose=1

# epochs = 6
#
save_checkpoints=False


print('running for '+str(epochs)+' epochs')




if config.local_or_cluster:
    # directory_name = '18-06-09-15-46-19'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = 'cnn'
else:
    # directory_name = '18-06-09-15-46-19'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = os.path.basename(__file__)

print('running '+directory_name+' '+file_name)

# Model parameters
activation = 'elu'
conv2_size = 2 #conv1_size=3
pool_size = (3, 1)
stride_size = (2, 1)
padding = 'same' #same: classic, you go until end of zero pad, valid: you truncate end if it doesnt fit in filter.
filter_sizes = [3] #[3,4,7] #Takes up to three.
batch_size = config.batch_size
# sequence_length = x.shape[1] # 56
sequence_length = config.sequence_length
embedding_dim = 300

def get_output(model, layer_name, batch_size=batch_size, Xvalidation=None, layer_2d_or_1d='2d'):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = intermediate_layer_model.predict(Xvalidation, batch_size=batch_size, verbose=verbose)
    if layer_2d_or_1d=='2d':
        layer_output = np.reshape(layer_output,(len(Xvalidation), int(layer_output.shape[1])*int(layer_output.shape[3])))
    layer_output = pd.DataFrame(layer_output)
    return layer_output

# layer_name = 'kmaxpool_1'
# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output_shape)
# Load data
##============================================================================================
#

# importlib.reload(data_helpers)

# Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=False) # I changed this so it combines train and validation
# Xvalidation, Yvalidation = data_helpers.load_data(config.validation_path, categories)
# Xvalidation_raw, Yvalidation_raw = data_helpers.load_data_raw(config.validation_path, categories)

# Xtrain, Ytrain = data_helpers.load_data(config.train_path, categories)
Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=False)
Xvalidation, Yvalidation = data_helpers.load_data(config.validation_path, categories)
Xvalidation_raw, Yvalidation_raw = data_helpers.load_data_raw(config.validation_path, categories)

## Encode Ytrain
# =====================================================================================
#one hot encode and integer encode
Ytrain_encoded = np_utils.to_categorical(Ytrain)
Ytrain_integer = np.array(Ytrain)
Yvalidation_encoded = np_utils.to_categorical(Yvalidation)
Yvalidation_integer = np.array(Yvalidation)

# Zero pad (encode) Xtrain and Xvalidation
# ==================================================================================================
tokenizer = Tokenizer(filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~') #TODO depending on word embedding, set lower=False.
tokenizer.fit_on_texts(np.append(np.array(Xtrain), np.array(Xvalidation)))
# tokenizer.fit_on_texts(Xtrain)
sequences = tokenizer.texts_to_sequences(Xtrain)
sequences2 = tokenizer.texts_to_sequences(Xvalidation)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Xtrain_encoded = pad_sequences(sequences, maxlen=sequence_length, padding='post')
Xvalidation_encoded = pad_sequences(sequences2, maxlen=sequence_length, padding='post')


#define max length of doc
# l1 = []
# for text in Xtrain:
#     l1.append(len(text_to_word_sequence(text)))

# def roundup(x):
#     return int(math.ceil(x / 100.0)) * 100
#
# sequence_length = roundup(np.max(l1)) #


## embedding layer https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# =====================================================================================

# Creating the model
#
# from gensim.models.keyedvectors import KeyedVectors
# it_model = KeyedVectors.load_word2vec_format(config.word_embeddings_path+'wiki.it.vec')
#
# # Getting the tokens
# words = []
# for word in it_model.vocab:
#     words.append(word)
#
# # Printing out number of tokens available
# print("Number of Tokens: {}".format(len(words)))
# # Printing out the dimension of a word vector
# print("Dimension of a word vector: {}".format(len(it_model[words[0]])))
# # Print out the vector of a word
# embeddings_index = {}
# for i in range(len(words)):
#     embeddings_index[words[i]] = it_model[words[i]]
#
# def save_obj(obj, path_and_filename):
#     with open(path_and_filename + '.pkl', 'wb+') as f:
#         pickle.dump(obj, f)
#
# save_obj(embeddings_index,'/Users/danielmlow/Dropbox/cnn/data/wiki.it/gensim_it_w2v')


def load_obj(path_and_filename):
    with open(path_and_filename, 'rb') as f:
        return pickle.load(f)


embeddings_index = load_obj(config.word_embeddings_path+'/gensim_it_w2v.pkl') #dictionary embeddings_index.get('è') returns word embedding

# from scipy import spatial
# res = spatial.distance.cosine(embeddings_index.get('uno'), embeddings_index.get('dieci'))

# embeddings_index = {}
# # with open(os.path.join(config.word_embeddings_path,'GoogleNews-vectors-negative300.bin')) as f:
# with open(os.path.join(config.word_embeddings_path,'glove.6B.'+str(embedding_dim)+'d.txt')) as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#
# print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim)) #this will be all embeddings for my vocabulary

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# # leverage our embedding_index dictionary and our word_index to compute our embedding matrix
# # ============================================================================================================
# embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
#
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#






path_to_dir = os.path.join(config.save_to, directory_name + '/')
try: os.makedirs(path_to_dir)
except: pass
print('directory_name: '+directory_name)
print('path_to_dir: '+path_to_dir)



# Model
## ======================================================================================================
print("Creating Model...")

def cnn(X, y, drop, batch_size, optimizer, num_filters, activation1, activation2, dense_1_neurons, dense_final_neurons, dense_2_layer):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=len(word_index) + 1, weights=[embedding_matrix],output_dim=embedding_dim, input_length=sequence_length,trainable=True)(inputs)
    dropout_1 = Dropout(drop)(embedding)
    reshape1 = Reshape((sequence_length,embedding_dim,1))(dropout_1)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),use_bias=True,  strides=1, padding='valid', activation=activation1, name='conv_1')(reshape1)
    maxpool_1 = MaxPool2D(pool_size=pool_size, strides=stride_size, padding=padding,name='pool_1')(conv_1)
    maxpool_1_reshape = Reshape( (int(maxpool_1.shape[1]),int(maxpool_1.shape[3]),1), name='pool_1_reshaped')(maxpool_1)
    dropout_2 = Dropout(drop)(maxpool_1_reshape)
    conv_2 = Conv2D(num_filters*2, kernel_size=(conv2_size, int(maxpool_1_reshape.shape[2])), padding='valid', kernel_initializer='normal', activation=activation1, name='conv_2')(dropout_2)
    maxpool_2 = MaxPool2D(pool_size=pool_size, strides=stride_size, padding='valid',name='pool_2')(conv_2)
    maxpool_2_reshape = Reshape((int(maxpool_2.shape[1]),int(maxpool_2.shape[3]),1),name='pool_2_reshaped')(maxpool_2)

    flatten = Flatten()(maxpool_2_reshape)
    dropout_3 = Dropout(drop)(flatten)
    dense_1 = Dense(units=dense_1_neurons, activation=activation2,name='dense_1')(dropout_3)
    dropout_4 = Dropout(drop)(dense_1)
    if dense_2_layer == 'yes':
        dense_2 = Dense(units=dense_1_neurons, activation=activation2, name='dense_2')(dropout_4)
        dropout_5 = Dropout(drop)(dense_2)
        dense_final = Dense(units=512, activation=activation2, name='dense_final')(dropout_5)
    else:
        dense_final = Dense(units=dense_final_neurons, activation=activation2, name='dense_final')(dropout_4)
    dropout_6 = Dropout(drop)(dense_final)
    softmax_final = Dense(units=len(categories), activation='softmax', name='softmax_final')(dropout_6)
    model = Model(inputs=inputs, outputs=softmax_final)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=batch_size, epochs=epochs,
                        verbose=verbose)  # starts training
    return model

with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write(file_name + '\n')
    f.write(directory_name+ '\n\n')


# dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
# batch_sizes = [64,128, 256,512]
# optimizers = ['RMSprop','Adam']
# activations2 = ['elu','relu']
# num_filters = 64
# dense_1_neurons = 1024
# dense_final_neurons = 64

# num_filters_all = [16,32,64,128]
# dense_1_neurons_all= [2048,1024]
# dense_2_layer_all = ['yes', 'no']
# dense_final_neurons_all = [128]
# drop = 0.2
# batch_size=256
# optimizer = 'Adam'
# activation1 = 'elu'
# activation2 = 'elu'


drop = 0.2
batch_size = 512
optimizer = 'Adam'
activation2 = 'elu'
activation1 = 'elu'
num_filters = 128
dense_1_neurons = 512
dense_final_neurons = 128
dense_2_layer = 'no'

from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(Xtrain_encoded, Ytrain)
print(skf)
scores = []
for train_index, test_index in skf.split(Xtrain_encoded, Ytrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xtrain_encoded[train_index], Xtrain_encoded[test_index]
    y_train, y_test = np.array(Ytrain)[train_index], np.array(Ytrain)[test_index]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    model = cnn(X_train, y_train, drop, batch_size, optimizer, num_filters, activation1, activation2, dense_1_neurons, dense_final_neurons, dense_2_layer)
    accuracy = model.evaluate(X_test, y_test,verbose=verbose)
    scores.append(accuracy[1])
    # print(i)
    with open(path_to_dir + 'log.txt', 'a+') as f:
        f.write(str(i)+'=================\n')
        # f.write('Parameters: '+str(parameters)+'\n')
        f.write('Loss and Accuracy: ' +str(accuracy)+'\n')
        f.write(str(np.round(accuracy[1],4)) + '\n\n')
    i += 1


with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write(str(scores))
    f.write('Accuracy: '+str(np.round(np.mean(scores),4))+' +/-  '+str((np.std(scores)* 2).round(2))+ '\n')
    f.write('SD: '+str(np.std(scores))+' variance: '+str(np.var(scores))+ '\n')





# l=[]
# for drop in dropout_rates:
#     for batch_size in batch_sizes:
#         for optimizer in optimizers:
#             for activation2 in activations2:
#                 l.append([drop, batch_size, optimizer,activation2])
# l[]
# l=[]
# for num_filters in num_filters_all:
#     for dense_1_neurons in dense_1_neurons_all:
#         for dense_final_neurons in dense_final_neurons_all:
#             for dense_2_layer in dense_2_layer_all:
#                 l.append([num_filters, dense_1_neurons, dense_final_neurons, dense_2_layer])
#
#
# gs_numb = int(sys.argv[1])
# i = int(sys.argv[1])
# print(i)
# for parameters in l[gs_numb:gs_numb+6]:
#     print(parameters)
#     # drop, batch_size, optimizer, activation2 = parameters
#     num_filters, dense_1_neurons, dense_final_neurons, dense_2_layer = parameters
#     model = cnn(drop, batch_size, optimizer, num_filters, activation1, activation2, dense_1_neurons, dense_final_neurons, dense_2_layer)
#     accuracy = model.evaluate(Xvalidation_encoded, Yvalidation_encoded,verbose=verbose)
#     print(i)
#     with open(path_to_dir + 'log.txt', 'a+') as f:
#         f.write(str(i)+'=================\n')
#         f.write('Parameters: '+str(parameters)+'\n')
#         f.write('Loss and Accuracy: ' +str(accuracy)+'\n')
#         f.write(str(np.round(accuracy[1],4)) + '\n\n')
#     i += 1





# if save_checkpoints:
#     filepath = path_to_dir+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" #https://machinelearningmastery.com/check-point-deep-learning-models-keras/
#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=verbose, save_best_only=True, mode='auto')
#
# print("Training Model...")
# if save_checkpoints:
#     history = model.fit(Xtrain_encoded, Ytrain_encoded, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[checkpoint])  # starts training
# else:
#     history = model.fit(Xtrain_encoded,Ytrain_encoded, batch_size=batch_size, epochs=epochs, verbose=verbose)  # starts training

# outputs:
# ============================================================================================================================
# SAVE




# model.save(path_to_dir + 'model.h5', overwrite=True) #Save model #TODO: not working nor checkpoint save model
# model.save_weights(path_to_dir +"model_weights.h5", overwrite=True)
# # plot_model(model, to_file=path_to_dir + 'model.png') # Save plot of model
# np.save(path_to_dir + 'history_dict.npy', history.history) #Save history
# plot_outputs.learning_curve(history.history, path_to_dir) #'loss' 'both'
# accuracy = model.evaluate(Xvalidation_encoded,Yvalidation_encoded,verbose=verbose)  # TODO:change to validation set for final model.
# Ypredict = model.predict(Xvalidation_encoded, batch_size=batch_size, verbose=verbose)  # TODO:change to validation set for final model.
# Ypredict_encoded = np_utils.to_categorical(Ypredict.argmax(axis=-1))
# Ypredict_integer = Ypredict.argmax(axis=-1)
# np.save(path_to_dir+'Ypredict_integer',Ypredict_integer)
# clas_rep = classification_report(Yvalidation_encoded, Ypredict_encoded,target_names=categories)  # TODO:change to validation set for final model.
# df_clas_rep, df_clas_rep_latex = plot_outputs.classification_report_df(clas_rep)
# cm = confusion_matrix(y_true=Yvalidation_integer, y_pred=Ypredict_integer,sample_weight=None)  # TODO:change to validation set for final model
# pd.DataFrame(cm, columns=categories, index=categories).to_csv(path_to_dir+'cm.csv')
# index_min = df_clas_rep['f1_score'].idxmin()
# index_max = df_clas_rep['f1_score'].idxmax()
#
# a2 = conv_1.shape[:]
# a2 = str(a2[1])+'*'+str(a2[3])+'='+str(a2[1]*a2[3])
# a3 = maxpool_1.shape[1:3]
# a3 = str(a3[0])+'*'+str(a3[1])+'='+str(a3[0]*a3[1])
# a4 = conv_2.shape[:]
# a4 = str(a4[1])+'*'+str(a4[3])+'='+str(a4[1]*a4[3])
# a5 = maxpool_2.shape[1:3]
# a5 = str(a5[0])+'*'+str(a5[1])+'='+str(a5[0]*a5[1])
#
# reporting = [len(categories),sequence_length,embedding.shape[1:],a2,a3,a4,a5,[''], dense_final_neurons,[''],activation, num_filters, filter_sizes,padding, pool_size, stride_size, drop, epochs, [''], df_clas_rep['f1_score'][len(categories)], [df_clas_rep['class'][index_min],df_clas_rep['f1_score'][index_min]], [df_clas_rep['class'][index_max],df_clas_rep['f1_score'][index_max]]]
# # reporting = [len(categories),sequence_length,embedding.shape[1:], dense_final_neurons,[''],activation, num_filters, filter_sizes,padding, pool_size, stride_size, drop, epochs, [''], df_clas_rep['f1_score'][len(categories)], [df_clas_rep['class'][index_min],df_clas_rep['f1_score'][index_min]], [df_clas_rep['class'][index_max],df_clas_rep['f1_score'][index_max]]]
#
# with open(path_to_dir + 'log.txt', 'a+') as f:
#     f.write(file_name+'\n')
#     f.write(directory_name)
#     f.write('\n\n')
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
#     f.write('\n\n')
#     for i, name in enumerate(model.metrics_names):
#         f.write(name+': '+str(np.round(accuracy[i], 2))+'\n')
#     # f.write('accuracy: %f ' % (accuracy * 100))
#     # f.write('loss: %f ' % (loss))
#     f.write('\n\n')
#     f.write('Classification Report: \n'+df_clas_rep_latex)
#     for i in reporting:
#         f.write(str(i)+'\n')

# # Save output_layers
# # loaded_model = load_model(path_to_dir+'model.h5')
# # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# conv_1 = get_output(model, 'conv_1', layer_2d_or_1d='2d', Xvalidation=Xvalidation_encoded)
# # kmaxpool_1 = get_output(model, 'kmaxpool_1', layer_2d_or_1d='2d', Xvalidation=Xvalidation_encoded)
# pool_1 = get_output(model, 'pool_1', layer_2d_or_1d='2d', Xvalidation=Xvalidation_encoded)
# conv_2 = get_output(model, 'conv_2', layer_2d_or_1d='2d', Xvalidation=Xvalidation_encoded)
# pool_2 = get_output(model, 'pool_2', layer_2d_or_1d='2d', Xvalidation=Xvalidation_encoded)
# dense_1 = get_output(model, 'dense_1', layer_2d_or_1d='1d', Xvalidation=Xvalidation_encoded)
# dense_final = get_output(model, 'dense_final', layer_2d_or_1d='1d', Xvalidation=Xvalidation_encoded)
# softmax_final = get_output(model, 'softmax_final', layer_2d_or_1d='1d', Xvalidation=Xvalidation_encoded)
#
# # np.savez_compressed(path_to_dir+'output_layers.npz', a=conv_1, b=pool_1,c=dense_general,d=conv_2, e=pool_2, f=dense_final)
# np.savez_compressed(path_to_dir+'output_layers.npz', a=conv_1, b=pool_1,c=conv_2, d=pool_2, e=dense_1, f=dense_final, g=softmax_final)
# # np.savez_compressed(path_to_dir+'output_layers.npz', a=conv_1, b=kmaxpool_1,c=conv_2, d=pool_2, e=dense_1, f=dense_final)
#
# # import importlib
# # importlib.reload(corr_between_layers)
#
# # Log.txt
# models = [directory_name]
# statistic = 'spearman'
#
# # layers= data_helpers.load_output_layers(path_to_dir)
# layers = [conv_1, pool_1, conv_2, pool_2, dense_1, dense_final, softmax_final]
#
# zeros1 = []
# zeros2 = []
# zeros3 = []
#
# for layer in layers:
#     zero1 = (np.count_nonzero(layer.round(1) == 0)) / (layer.shape[0] * layer.shape[1])
#     zeros1.append(np.round(zero1,3))
#     zero2 = (np.count_nonzero(layer.round(2) == 0)) / (layer.shape[0] * layer.shape[1])
#     zeros2.append(np.round(zero2, 3))
#     zero3 = (np.count_nonzero(layer.round(3) == 0)) / (layer.shape[0] * layer.shape[1])
#     zeros3.append(np.round(zero3, 3))
#
# with open(path_to_dir + 'log.txt', 'a+') as f:
#     f.write('zeros round to 1 decimal: '+str(zeros1)+'\n')
#     f.write('zeros round to 2 decimals: '+str(zeros2)+'\n')
#     f.write('zeros round to 3 decimals: '+str(zeros3)+'\n')
#
#
# # corr_between_layers1 = corr_between_layers.corr_between_layers_method(statistic,models,layers, config.save_to,Xvalidation, categories, model_type=file_name)
# # reporting.append(list(np.round(corr_between_layers1.iloc[-1, :-1], 2)))
#
#
# # ===================================================================================================================
# import random
# # Generate random sequence so you always compare same sentences
# amount_sent = int(len(Xvalidation)/len(categories))
# # sentences_index = list(range(amount_sent))
# # random.shuffle(sentences_index)
# sentences_index = [767, 445, 551, 566, 224, 853, 836, 564, 575, 793, 730, 759, 82, 390, 110, 804, 615, 577, 815, 290, 190, 239, 255, 731, 851, 186, 799, 628, 940, 209, 580, 13, 533, 926, 125, 2, 285, 784, 130, 171, 181, 599, 500, 457, 432, 753, 847, 118, 795, 570, 729, 56, 769, 929, 701, 64, 387, 687, 144, 33, 325, 423, 828, 657, 166, 96, 175, 664, 671, 87, 9, 315, 204, 741, 57, 588, 245, 407, 433, 326, 902, 820, 916, 187, 416, 882, 357, 105, 693, 349, 520, 501, 676, 698, 737, 267, 910, 899, 327, 362, 320, 478, 309, 198, 346, 824, 734, 678, 529, 523, 240, 652, 884, 351, 356, 89, 462, 283, 466, 116, 287, 258, 833, 219, 591, 381, 74, 900, 915, 241, 37, 4, 430, 605, 498, 141, 220, 494, 376, 601, 849, 379, 688, 506, 644, 164, 864, 424, 414, 957, 422, 810, 686, 363, 92, 233, 102, 595, 717, 380, 579, 908, 126, 297, 319, 496, 228, 733, 52, 524, 377, 0, 221, 632, 199, 8, 642, 256, 858, 143, 119, 299, 821, 145, 448, 848, 611, 188, 817, 463, 403, 368, 140, 685, 700, 806, 68, 935, 567, 919, 619, 440, 742, 123, 232, 114, 608, 23, 76, 304, 903, 775, 932, 259, 493, 720, 948, 223, 251, 542, 340, 77, 237, 460, 665, 812, 360, 639, 640, 323, 749, 450, 578, 782, 208, 196, 863, 773, 797, 475, 625, 331, 288, 850, 61, 563, 798, 139, 527, 479, 316, 277, 525, 402, 12, 598, 722, 473, 149, 574, 101, 162, 860, 586, 172, 715, 941, 272, 616, 662, 270, 62, 585, 375, 373, 807, 339, 877, 46, 173, 284, 526, 163, 825, 248, 594, 857, 866, 622, 808, 868, 497, 572, 231, 88, 249, 117, 67, 518, 881, 573, 788, 385, 453, 956, 435, 874, 544, 438, 839, 593, 311, 505, 394, 568, 545, 943, 904, 781, 613, 442, 41, 246, 648, 179, 276, 292, 397, 787, 689, 834, 182, 38, 42, 274, 582, 345, 94, 928, 597, 242, 732, 111, 511, 534, 716, 726, 624, 324, 337, 875, 756, 571, 160, 294, 289, 951, 441, 65, 831, 467, 286, 206, 790, 225, 418, 404, 413, 282, 446, 508, 661, 39, 569, 250, 663, 18, 637, 485, 837, 539, 461, 895, 213, 439, 675, 543, 630, 658, 487, 855, 491, 476, 822, 838, 587, 378, 561, 924, 28, 865, 44, 428, 826, 7, 727, 844, 384, 469, 654, 60, 829, 348, 763, 152, 672, 695, 449, 230, 764, 128, 938, 129, 301, 452, 426, 709, 32, 590, 159, 631, 918, 192, 528, 34, 897, 468, 612, 19, 40, 547, 516, 482, 408, 194, 382, 623, 330, 106, 419, 168, 703, 207, 437, 724, 465, 431, 776, 271, 536, 75, 170, 934, 670, 744, 275, 85, 15, 358, 841, 81, 515, 254, 553, 667, 842, 679, 750, 71, 765, 43, 655, 712, 805, 819, 761, 811, 191, 614, 310, 937, 36, 723, 950, 21, 883, 161, 747, 169, 321, 263, 214, 336, 714, 830, 659, 617, 552, 51, 954, 406, 10, 234, 474, 183, 146, 405, 603, 6, 484, 581, 458, 873, 604, 522, 770, 328, 636, 861, 156, 548, 707, 499, 746, 813, 354, 257, 222, 11, 393, 708, 540, 684, 79, 618, 401, 133, 713, 803, 789, 22, 876, 489, 927, 153, 682, 268, 70, 791, 920, 556, 681, 303, 786, 513, 921, 792, 706, 913, 507, 852, 260, 280, 54, 645, 412, 420, 100, 361, 53, 415, 115, 308, 127, 514, 886, 565, 471, 896, 892, 800, 859, 740, 490, 939, 584, 945, 226, 592, 949, 823, 26, 550, 338, 131, 602, 626, 350, 878, 854, 953, 470, 495, 748, 777, 367, 210, 683, 641, 562, 158, 492, 55, 486, 344, 421, 802, 785, 372, 136, 107, 73, 690, 151, 455, 725, 669, 879, 643, 780, 90, 366, 719, 621, 923, 84, 147, 371, 177, 306, 692, 589, 610, 867, 779, 743, 383, 634, 893, 31, 554, 137, 503, 649, 718, 174, 193, 391, 5, 334, 755, 898, 946, 197, 298, 216, 176, 739, 370, 930, 472, 889, 751, 609, 78, 291, 558, 124, 318, 754, 48, 521, 512, 705, 809, 157, 396, 653, 203, 447, 300, 178, 801, 880, 451, 132, 47, 353, 925, 399, 417, 646, 436, 342, 794, 890, 481, 333, 738, 796, 165, 955, 244, 647, 952, 409, 905, 721, 80, 771, 121, 313, 189, 517, 870, 596, 103, 329, 185, 425, 656, 3, 359, 560, 894, 519, 374, 99, 541, 480, 355, 269, 1, 135, 783, 205, 235, 843, 835, 58, 25, 247, 530, 108, 236, 215, 252, 410, 386, 633, 17, 50, 483, 958, 557, 936, 607, 832, 914, 389, 93, 760, 818, 735, 150, 872, 218, 906, 211, 459, 650, 942, 752, 546, 532, 454, 699, 762, 278, 745, 72, 138, 279, 704, 14, 109, 364, 766, 200, 395, 69, 302, 535, 758, 83, 184, 907, 846, 917, 862, 365, 434, 660, 97, 728, 95, 845, 606, 901, 668, 576, 531, 774, 217, 265, 314, 477, 195, 600, 909, 305, 444, 694, 322, 227, 86, 559, 549, 29, 398, 510, 91, 266, 45, 167, 814, 201, 295, 154, 388, 509, 347, 816, 651, 538, 148, 691, 180, 261, 343, 638, 155, 400, 122, 456, 891, 49, 443, 293, 711, 736, 944, 464, 933, 772, 113, 24, 212, 59, 243, 332, 666, 710, 312, 768, 66, 273, 238, 856, 696, 296, 335, 427, 134, 635, 778, 63, 502, 317, 112, 35, 98, 827, 307, 20, 429, 697, 947, 488, 341, 702, 680, 264, 871, 620, 911, 352, 583, 202, 757, 262, 885, 673, 504, 887, 677, 411, 16, 912, 30, 922, 392, 369, 27, 229, 120, 281, 840, 142, 537, 555, 931, 629, 627, 959, 253, 888, 869, 104, 674]
#
# def corr_layers(n=None, layers=None, layer_names=None, statistic='spearman'):
#     corr_matrices = []
#     a = np.zeros((len(layers),len(layers)))
#     corr_between_layers = pd.DataFrame(a, columns=layer_names, index=layer_names)
#     # obtain N sentences from each category
#     for layer in layers:
#         randomN_sentences = pd.DataFrame()
#         Nsentences = sentences_index[:n]
#         for i in range(len(categories)):
#             category_start_number = i*amount_sent
#             Nsentences_1category= [n+category_start_number for n in Nsentences]
#             randomN_sentences_1category = layer.iloc[Nsentences_1category, :]
#             randomN_sentences = pd.concat([pd.DataFrame(randomN_sentences), randomN_sentences_1category],axis=0)
#         corr_matrix_lX = randomN_sentences.T.corr(method=statistic)
#         corr_matrix_lX_triu = pd.DataFrame(np.triu(corr_matrix_lX, k=1)).replace(0, np.nan)
#         corr_matrices.append(corr_matrix_lX_triu)
#     for j in range(0,len(layers)):
#         for i in range(0,len(layers)):
#             res = corr_matrices[j].corrwith(corr_matrices[i]).mean() #TODO: Pearson. correlation between layers. # res = corr_matrices[j].apply(lambda col: col.corr(corr_matrices[i], method=statistic), axis=0)
#             corr_between_layers.iloc[j,i] = res
#     return corr_between_layers
#
# statistic='spearman'
# sentences = 40
# # layer_names = ['lstm1', 'dense1', 'dense2', 'dense_final', 'softmax_final']
# layer_names = ['conv_1', 'pool_1', 'conv_2', 'pool_2', 'dense_1', 'dense_final', 'softmax_final']
# layers = [conv_1, pool_1, conv_2, pool_2, dense_1, dense_final, softmax_final]
# corr_between_layers1 = corr_layers(n=sentences, layers=layers, layer_names=layer_names)
# random.shuffle(sentences_index)
# corr_between_layers2 = corr_layers(n=sentences, layers=layers, layer_names=layer_names)
#
# with open(path_to_dir + 'log.txt', 'a+') as f:
#     f.write('\nCorrelation between layers' +str(sentences)+ 'sentences (spearman)\n')
#     f.write(str(corr_between_layers1.round(2)) + '\n')
#     f.write('\nCorrelation between layers' +str(sentences)+ 'different sentences (spearman)\n')
#     f.write(str(corr_between_layers2.round(2)) + '\n')
#     # f.write('\nCorrelation between layers (kendall)\n')
#     # f.write(str(corr_between_layers2.round(2)) + '\n')
#     # f.write('\nCorrelation between layers (kendall)\n')
#     # f.write(str(corr_between_layers2.round(2)) + '\n')
#
# corr_between_layers0 = pd.concat([corr_between_layers1.round(2), corr_between_layers2.round(2)])
# corr_between_layers0.to_csv(path_to_dir+'corr_between_layers0.csv', index=True, header=True)
#
#
#
#
# # path_to_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/cnn10/'
# # RSA_arr_dense_final = np.load(path_to_dir+'RSA_arr_dense_final.npy')
# # layer_name='dense_finalb'
# #
# # from scipy.spatial import distance
# # from scipy.cluster import hierarchy
# # import seaborn as sns
# # # correlations = df.corr()
# # # correlations_array = np.asarray(df.corr())
# #
# # correlations = pd.DataFrame(RSA_arr_dense_final, columns=categories, index=categories)
# # correlations_array = RSA_arr_dense_final[:]
# #
# # row_linkage = hierarchy.linkage(
# #     distance.pdist(correlations_array, metric='euclidean'), method='ward', optimal_ordering=True)
# #
# # col_linkage = hierarchy.linkage(
# #     distance.pdist(correlations_array.T, metric='euclidean'), method='ward', optimal_ordering=True)
# # #
# # # sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=col_linkage, row_colors=network_colors, method="average",
# # #                col_colors=network_colors, figsize=(13, 13), cmap=cmap)
# # sns.set(font_scale=0.5)
# # cg = sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=col_linkage,cmap="RdBu_r", vmin = -0.8, vmax=0.8, cbar_kws={"ticks":[-0.8,-0.4,0.0, 0.4, 0.8]})
# # plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# # plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# # cg.savefig(path_to_dir + 'RSA_ward_'+ layer_name + '.eps', format='eps', dpi=100)
# #
# # rsa.plot_rsm(path_to_dir, RSA_arr_dense_final, categories, layer_name='dense_finalb')
#
# conv_1_shape = sequence_length - filter_sizes[0] + 1
# pool_1_shape = int(conv_1_shape/stride_size[0])
# #
# # # TODO: choose short and long sentence from new dataset.
# #
# # for i in range(len(Xvalidation)):
# #     length = len(Xvalidation[i].split())
# #     if length == 20:
# #         print(i)
# #         break
#
# # 51 has 10 words, 10 has 20 words
# for sentence_id in [51]:
#     for layer, layer_name in zip(layers[:2], layer_names[:2]):
#         plt.clf()
#         sentence_length = len(Xvalidation[sentence_id].split())
#         sentence = np.array(layer.T[sentence_id])
#         sns.set(font_scale=0.5)
#         if layer_name =='conv_1':
#             reshaped  = np.reshape(sentence, [conv_1_shape, num_filters])
#         elif layer_name =='pool_1':
#             reshaped = np.reshape(sentence, [pool_1_shape, num_filters])
#         elif layer_name =='conv_2':
#             reshaped = np.reshape(sentence, [pool_1_shape-2, num_filters*2])
#         elif layer_name =='pool_2':
#             reshaped = np.reshape(sentence, [int((pool_1_shape-2)/2), num_filters*2])
#         else:
#             reshaped = np.reshape(sentence, [1,layer.shape[1]])
#         cg = sns.heatmap(reshaped, cmap="RdBu_r", vmin=-1., vmax=1.,
#                             cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
#         # cg = sns.heatmap(reshaped, cmap="RdBu_r")
#         # cg = sns.heatmap(reshaped, cmap="Reds")
#         plt.xticks(rotation=90)
#         plt.ylabel('Words')
#         plt.yticks(rotation=0)
#         plt.xlabel('Filters/Channels')
#         plt.title(layer_name+' - Single Sentence \n')
#         plt.savefig(path_to_dir+'single_sent_heatmap_'+layer_name+'_'+str(sentence_length)+'words.png')
#
# # TODO: careful when using layer[-1] because I want to use dense_final[-2] not softmax_final [-1]
# df_prototypical, df_prototypical_score, df_prototypical_sentences = corr_between_layers.prototypical_sentences(statistic, Xvalidation,Xvalidation_raw, path_to_dir, layer=layers[-2], validation_size= len(Xvalidation), amount_sent=int(len(Xvalidation)/len(categories)),nan='', categories=categories)
#
# if plot_RSA:
#     # rsa.plot_RSA(path_to_dir, categories, layer=conv_1, layer_name='conv_1',
#     #              amount_sent=int(conv_1.shape[0] / len(categories)))
#     # rsa.plot_RSA(path_to_dir, categories, layer=pool_1, layer_name='pool_1',
#     #              amount_sent=int(conv_1.shape[0] / len(categories)))
#     # rsa.plot_RSA(path_to_dir, categories, layer=conv_2, layer_name='conv_2', amount_sent=int(conv_1.shape[0]/len(categories)))
#     # rsa.plot_RSA(path_to_dir, categories, layer=pool_2, layer_name='pool_2',
#     #              amount_sent=int(conv_1.shape[0] / len(categories)))
#     rsa.plot_RSA(path_to_dir, categories, layer=dense_1, layer_name='dense_1', amount_sent=int(conv_1.shape[0]/len(categories)))
#     rsa.plot_RSA(path_to_dir, categories, layer=dense_final, layer_name='dense_final',
#                  amount_sent=int(len(Xvalidation) / len(categories)))
#     rsa.plot_RSA(path_to_dir, categories, layer=softmax_final, layer_name='softmax_final',
#                  amount_sent=int(len(Xvalidation)/ len(categories)))
#     # rsa.plot_RSA(path_to_dir, categories, layer=dense_general, layer_name='dense_general',amount_sent=int(conv_1.shape[0] / len(categories)))
#     # rsa.plot_RSA(path_to_dir, categories, layer=softmax_final_sigmoid, layer_name='softmax_final_sigmoig')
#     # rsa.plot_RSA(path_to_dir, categories, layer=softmax_final_elu, layer_name='softmax_final_elu')
#
#
# # def Nsentences_a_k_length(n,a, k, layer, amount_sent_per_category, Xvalidation):
# #     layer_sample = pd.DataFrame()
# #     for i in range(0,layer.shape[0], amount_sent_per_category):
# #         n_sentences_same_length = pd.DataFrame()
# #         sentences = 0
# #         # loop through sentences in categories
# #         for j in range(i,i+amount_sent_per_category):
# #             if (len(Xvalidation[j].split())>=a & len(Xvalidation[j].split())<=k):
# #                 n_sentences_same_length = n_sentences_same_length.append(layer.T[j])
# #                 sentences+=1
# #                 if sentences == n:
# #                     print('a category does not have enough')
# #                     break
# #         if n_sentences_same_length.shape[0] < n:
# #             print('not enough sentences of that length. Try again.')
# #             break
# #         else:
# #             layer_sample = layer_sample.append(n_sentences_same_length)
# #     return layer_sample
#
#
# def Nsentences(n, layer, amount_sent_per_category):
#     layer_sample = pd.DataFrame()
#     for i in range(0,layer.shape[0], amount_sent_per_category):
#         n_sentences_1category = layer.iloc[i:i+amount_sent_per_category]
#         n_sentences_1category = n_sentences_1category.sample(frac=1).iloc[:n]
#         layer_sample = layer_sample.append(n_sentences_1category)
#     return layer_sample
#
# # RSA single sentences
# # =====================================================================================================================================
# amount_sent_per_category = int(len(Xvalidation)/len(categories))
# n = 5
#
# layer_names = ['conv_1', 'pool_1', 'conv_2', 'pool_2','dense_1' ,'dense_final', 'softmax_final']
# for layer, layer_name in zip(layers, layer_names):
#     # layer_sample = Nsentences_a_k_length(n, 10, 14, layer, amount_sent_per_category, Xvalidation)
#     layer_sample = Nsentences(n, layer, amount_sent_per_category)
#     # Clustermap
#     df = pd.DataFrame(layer_sample)
#     # df[(df >= -0.08) & (df <= 0.09)] = np.nan
#     statistic = 'spearman'
#     df = df.T.corr(method=statistic)
#     columns = [[i] * n for i in categories]  # TODO: categories or categories_wrong.
#     # columns = [[i]*n for i in config.categories_wrong]
#     columns = [i for j in columns for i in j]
#     df.columns = columns
#     df.index = columns
#     sns.set(font_scale=0.08)
#     cg = sns.clustermap(df, method='ward', cmap="RdBu_r", vmin=-1., vmax=1.,cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
#     plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, )
#     plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#     # tick_locator = ticker.MaxNLocator(int(df.shape[0]))
#     # plt.setp(cg.ax_heatmap.xaxis.set_major_locator(tick_locator))
#     # plt.setp(cg.ax_heatmap.yaxis.set_major_locator(tick_locator))
#     cg.savefig(path_to_dir+ '/RSA_ward_clustermap_' + layer_name + '_single_sentences_'+statistic+'.eps', format='eps', dpi=100)
#
# # With NaNs just first two layers
# for layer, layer_name in zip(layers[:2], layer_names[:2]):
#     # layer_sample = Nsentences_a_k_length(n, 10, 14, layer, amount_sent_per_category, Xvalidation)
#     layer_sample = Nsentences(n, layer, amount_sent_per_category)
#     # Clustermap
#     df = pd.DataFrame(layer_sample)
#     df[(df >= -0.09) & (df <= 0.09)] = np.nan
#     statistic = 'spearman'
#     df = df.T.corr(method=statistic)
#     columns = [[i] * n for i in categories]  # TODO: categories or categories_wrong.
#     # columns = [[i]*n for i in config.categories_wrong]
#     columns = [i for j in columns for i in j]
#     df.columns = columns
#     df.index = columns
#     sns.set(font_scale=0.08)
#     try:
#         cg = sns.clustermap(df, method='ward', cmap="RdBu_r", vmin=-1., vmax=1.,
#                             cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
#         plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, )
#         plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#         # tick_locator = ticker.MaxNLocator(int(df.shape[0]))
#         # plt.setp(cg.ax_heatmap.xaxis.set_major_locator(tick_locator))
#         # plt.setp(cg.ax_heatmap.yaxis.set_major_locator(tick_locator))
#         cg.savefig(path_to_dir+ '/RSA_ward_clustermap_' + layer_name + '_single_sentences_'+statistic+'_with_NaNs.eps', format='eps', dpi=100)
#     except:
#         pass
#
#
# layer_names = ['conv_1_nans', 'pool_1_nans', 'conv_2_nans', 'pool_2_nans', 'dense_final_nans']
# layers_nans=[]
# for layer, layer_name in zip(layers, layer_names):
#     df = pd.DataFrame(layer)
#     df[(df>=-0.09) & (df<=0.09)] = np.nan
#     layers_nans.append(df)
#     with open(path_to_dir + 'log.txt', 'a+') as f:
#         f.write(layer_name+': amount of nans: ')
#         f.write(str(df.isnull().sum().sum())+ ' ')
#         f.write('rel. amount: '+str((df.isnull().sum().sum()/(df.shape[0]*df.shape[1])).round(2)))
#         f.write('\n')
#     # rsa.plot_RSA(path_to_dir, categories, layer=df, layer_name=layer_name, amount_sent=int(layer.shape[0]/len(categories)))
#
# #
# # corr_between_layers1 = corr_between_layers.corr_between_layers_method(statistic,models,layers_nans,config.save_to,Xvalidation, categories,nan='with_nans_', model_type=file_name)
# #
#
# # corr_between_layers1 = corr_layers(n_random_sentences, df_prototypical,layers, df_prototypical_sentences, layer_names, statistic, categories=categories)
#
# def Nsentences_Klength(n, sentence_length, layer, amount_sent_per_category, Xvalidation):
#     layer_sample = pd.DataFrame()
#     for i in range(0,layer.shape[0], amount_sent_per_category):
#         n_sentences_same_length = pd.DataFrame()
#         sentences = 0
#         # loop through sentences in categories
#         for j in range(i,i+amount_sent_per_category):
#             if len(Xvalidation[j].split())==sentence_length:
#                 n_sentences_same_length = n_sentences_same_length.append(layer.T[j])
#                 sentences+=1
#                 if sentences == n:
#                     break
#         if n_sentences_same_length.shape[0] < n:
#             print('not enough sentences of that length. Try again.')
#             break
#         else:
#             layer_sample = layer_sample.append(n_sentences_same_length)
#     return layer_sample
#
# for layer, layer_name in zip(layers[:2], layer_names[:2]):
#     amount_sent_per_category = int(len(Xvalidation)/len(categories))
#     n = 5
#     k = [8,28]
#     layer_sample_small = Nsentences_Klength(n, k[0], layer, amount_sent_per_category, Xvalidation)
#     layer_sample_large = Nsentences_Klength(n, k[1], layer, amount_sent_per_category, Xvalidation)
#     statistic = 'spearman'
#     sentences_from_each = len(categories)*n
#     layer_sample = pd.concat([layer_sample_small, layer_sample_large])
#     df = pd.DataFrame(layer_sample)
#     df = df.T.corr(method=statistic) #TODO: should I transpose like this?
#     df.index= range(0,sentences_from_each*2)
#     df.columns= range(0,sentences_from_each*2)
#     df_triu = pd.DataFrame(np.triu(df, k=1)).replace(0, np.nan)
#
#     zeros = pd.DataFrame(np.ones([sentences_from_each,sentences_from_each]))
#     a15 = pd.DataFrame(np.full([sentences_from_each, sentences_from_each], 20))
#     sentence_len1 = pd.concat([zeros, a15])
#     sentence_len2 = pd.concat([a15, zeros])
#     sentence_len = pd.concat([sentence_len1 ,sentence_len2], axis=1)
#     sentence_len.index=range(sentences_from_each*2)
#     sentence_len.columns=range(sentences_from_each*2)
#     sentence_len_triu= pd.DataFrame(np.triu(sentence_len, k=0)).replace(0, np.nan)
#     sentence_len_triu = sentence_len_triu.replace(1,0)
#     # zeros = pd.DataFrame(np.zeros([3,3]))
#     # a15 = pd.DataFrame(np.full([3, 3], 15 ))
#     # sentence_len.to_csv(output_dir+'sentence_len_distance_matrix.csv', header=False, index=False)
#     # sentence_len_triu.to_csv(output_dir+'sentence_len_distance_matrix.csv', header=False, index=False)
#
#     with open(path_to_dir + 'log.txt', 'a+') as f:
#         f.write('Sentence_len_triu_dist effect '+layer_name+' '+str(k)+': \n')
#         f.write(str(sentence_len_triu.corrwith(df_triu).mean())+'\n')
#         f.write(str(categories)+'\n')
#
#
#

