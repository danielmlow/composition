'''
This is based on cnn35_64. This is after the first pilot. 
Changes:
-don't filter out # in the tokenizer, tokenize both together. or save tokenizer https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
-use 'number' w2v as representation for any digit
-shuffling problem should be check before advancing: plot random selection of conv1 layers. theys should all be 14 or 15.
-tune hyperparameters. 
'''

from sklearn.linear_model import LogisticRegression
import time
import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.utils import np_utils
from numpy.random import seed
seed(123)

import os
import data_helpers
import config
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Parameters
# =====================================================================
categories = config.categories
verbose = config.verbose
save_checkpoints = config.save_checkpoints
plot_RSA = config.plot_RSA

if config.local_or_cluster:
    categories = categories[:3]
    epochs=1
    verbose=1
else:
    epochs = config.epochs  # it will probably need more.

# epochs = 6
#
save_checkpoints=False


print('running for '+str(epochs)+' epochs')




if config.local_or_cluster:
    # directory_name = '18-07-18-00-24-06'
    # directory_name = 'svm_gs0_test'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = 'logreg'
else:
    # directory_name = 'svm_gs0_test'
    # directory_name = '18-07-18-00-24-06'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = os.path.basename(__file__)

print('running '+directory_name+' '+file_name)


Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=False) # I changed this so it combines train and test
Xtest, Ytest = data_helpers.load_data(config.test_path, categories)
# Xtest_raw, Ytest_raw = data_helpers.load_data_raw(config.test_path, categories)
# X, y= data_helpers.load_whole_dataset(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)

# Remove Stopwords
# with open('stopwords-it2.txt', 'r') as f:
#     sw = f.readlines()
#
# italian_stop_words = [n.replace('\n','') for n in sw]
#
# Xtrain2 = []
# for sentence in Xtrain:
#     sentence_no_stopwords = ' '.join([word for word in sentence.split() if word not in italian_stop_words])
#     Xtrain2.append(sentence_no_stopwords)
#
# Xtest2 = []
# for sentence in Xtest:
#     sentence_no_stopwords = ' '.join([word for word in sentence.split() if word not in italian_stop_words])
#     Xtest2.append(sentence_no_stopwords)
# #
# # ## Encode Ytrain
# # # =====================================================================================
# # one hot encode and integer encode
# # Ytrain_encoded = np_utils.to_categorical(Ytrain)
# # Ytrain_integer = np.array(Ytrain)
# # Ytest_encoded = np_utils.to_categorical(Ytest)
# # Ytest_integer = np.array(Ytest)
# #
# # # Zero pad (encode) Xtrain and Xtest
# # # ==================================================================================================
# tokenizer = Tokenizer(filters='') #depending on word embedding, set lower=False.
# tokenizer.fit_on_texts(np.append(np.array(Xtrain2), np.array(Xtest2)))
# sequences = tokenizer.texts_to_sequences(Xtrain2)
# sequences2 = tokenizer.texts_to_sequences(Xtest2)
# # sequences3 = tokenizer.texts_to_sequences(X)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# # Xtrain_encoded = pad_sequences(sequences, maxlen=sequence_length, padding='post')
# # Xtest_encoded = pad_sequences(sequences2, maxlen=sequence_length, padding='post')
# # X_encoded = pad_sequences(sequences3, maxlen=sequence_length, padding='post')
#
# def load_obj(path_and_filename):
#     with open(path_and_filename, 'rb') as f:
#         return pickle.load(f)
#
# embeddings_index = load_obj(config.word_embeddings_path+'/gensim_it_w2v.pkl') #dictionary embeddings_index.get('è') returns word embedding
#
#
#
# number = np.random.normal(0., 0.23, 300)
#
# embedding_dim = 300
#
# embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))  # this will be all embeddings for my vocabulary
#
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#     elif "#" in word:
#         embedding_matrix[i] = number


# Create average sentence vectors

# Xtrain = []
# for sequence in sequences:
#     all_word_embeddings = []
#     for word_id in sequence:
#         embedding = embedding_matrix[word_id]
#         if np.sum(embedding)!=0.0:
#             all_word_embeddings.append(embedding)
#     mean_sentence_vector = list(pd.DataFrame(all_word_embeddings).mean())
#     if len(mean_sentence_vector)==0:
#         mean_sentence_vector = list(np.random.normal(0., 0.23, 300))
#     Xtrain.append(mean_sentence_vector)
#
# Xtest = []
# for sequence in sequences2:
#     all_word_embeddings = []
#     for word_id in sequence:
#         embedding = embedding_matrix[word_id]
#         all_word_embeddings.append(embedding)
#     mean_sentence_vector = list(pd.DataFrame(all_word_embeddings).mean())
#     if len(mean_sentence_vector)==0:
#         mean_sentence_vector = list(np.random.normal(0., 0.23, 300))
#     Xtest.append(mean_sentence_vector)



path_to_dir = os.path.join(config.save_to, directory_name + '/')
try: os.makedirs(path_to_dir)
except: pass
print('directory_name: '+directory_name)
print('path_to_dir: '+path_to_dir)

# Xtrain = np.array(Xtrain)
# Xtest = np.array(Xtest)
#
# np.save(config.word_embeddings_path+'Xtrain_w2v_mean', Xtrain)
# np.save(config.word_embeddings_path+'Xtest_w2v_mean', Xtest)

if config.local_or_cluster:
    amount_of_categories = len(categories)
    Xtrain = np.load(config.word_embeddings_path + 'Xtrain_w2v_mean.npy')
    Xtrain = [list(n) for n in Xtrain][:amount_of_categories*4800]
    Xtest = np.load(config.word_embeddings_path + 'Xtest_w2v_mean.npy')
    Xtest = [list(n) for n in Xtest][:amount_of_categories*1200]
else:
    Xtrain = np.load(config.word_embeddings_path+'Xtrain_w2v_mean.npy')
    Xtrain = [list(n) for n in Xtrain]
    Xtest = np.load(config.word_embeddings_path+'Xtest_w2v_mean.npy')
    Xtest = [list(n) for n in Xtest]


# Model
## ======================================================================================================
print("Creating Model...")

with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write(file_name + '\n')
    f.write(directory_name+ '\n\n')


# Cs = [0.01, 0.1, 1, 10]
# kernels = ['linear', 'rbf']
# kernels = ['linear']
# max_features_all = [100000,None]
# stop_words = [italian_stop_words, None]


# Final
# Top1 and Top5 accuracy on test set.

# clf = LinearSVC(verbose=verbose)

if config.local_or_cluster:
    Xtrain_toy = []
    for i in range(0,len(Xtrain), 100):
        Xtrain_toy.append(Xtrain[i])
    Ytrain_toy = []
    for i in range(0, len(Ytrain), 100):
        Ytrain_toy.append(Ytrain[i])
    Xtest_toy = []
    for i in range(0,len(Xtest), 100):
        Xtest_toy.append(Xtest[i])
    Ytest_toy = []
    for i in range(0, len(Ytest), 100):
        Ytest_toy.append(Ytest[i])

# start = time.time()
# clf = LogisticRegression(verbose=verbose)
# clf.fit(Xtrain, Ytrain)
# probs = clf.predict_proba(Xtest)

# best_1 = np.argsort(probs, axis=1)
# best_1 = [n[-1] for n in best_1]
# top1_accuracy = np.round(np.sum(np.array(Ytest)==np.array(best_1))/len(Ytest),4)
#
#
# best_2 = np.argsort(probs, axis=1)
# best_2 = [n[-2:] for n in best_2]
# top2_acc = []
# for i in range(len(best_2)):
#     if Ytest[i] in best_2[i]:
#         top2_acc.append(1)
#     else:
#         top2_acc.append(0)
#
# top2_accuracy = np.round(np.sum(top2_acc)/len(Ytest),4)
#
#
#
# best_3 = np.argsort(probs, axis=1)
# best_3 = [n[-3:] for n in best_3]
# top3_acc = []
# for i in range(len(best_3)):
#     if Ytest[i] in best_3[i]:
#         top3_acc.append(1)
#     else:
#         top3_acc.append(0)
#
# top3_accuracy = np.round(np.sum(top3_acc)/len(Ytest),4)
#
# best_5 = np.argsort(probs, axis=1)
# best_5 = [n[-5:] for n in best_5]
# top5_acc = []
# for i in range(len(best_5)):
#     if Ytest[i] in best_5[i]:
#         top5_acc.append(1)
#     else:
#         top5_acc.append(0)
#
# top5_accuracy = np.round(np.sum(top5_acc)/len(Ytest),4)
#
# # Ypredict_encoded = np_utils.to_categorical(Ypredict.argmax(axis=-1))
# # Ypredict_integer = Ypredict.argmax(axis=-1)
# # Save outputs
# np.save(path_to_dir + 'Ypredict_integer', best_1)
# np.save(path_to_dir + 'accuracy_integer_top3', top3_acc)
# np.save(path_to_dir + 'accuracy_integer_top5', top5_acc)
# np.save(path_to_dir + 'log_reg_coefficients', clf.coef_)
#
#
# with open(path_to_dir + 'log.txt', 'a+') as f:
#     end = time.time()
#     f.write(str(i)+'=================\n')
#     f.write('time: ' + str(np.round(end - start, 2))+'\n')
#     f.write('Top-1 Accuracy: ' +str(top1_accuracy)+'\n')
#     f.write('Top-2 Accuracy: ' + str(top2_accuracy) + '\n')
#     f.write('Top-3 Accuracy: ' + str(top3_accuracy) + '\n')
#     f.write('Top-5 Accuracy: ' + str(top5_accuracy) + '\n')
#     f.write(str(clf.get_params())+'\n\n')
#
# np.save(path_to_dir + 'probability_output', probs)

# # CV
# # ====================================================================
from sklearn.model_selection import cross_val_score

# start = time.time()
clf = LogisticRegression(verbose=verbose)
scores = cross_val_score(clf, Xtrain, Ytrain, cv=5)

with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write(str(scores))
    f.write('Accuracy: ' + str(np.round(np.mean(scores), 4)) + ' +/-  ' + str((np.std(scores) * 2).round(2)) + '\n')
    f.write('SD: ' + str(np.std(scores)) + ' variance: ' + str(np.var(scores)) + '\n')










# Gridsearch
# =============

#
# Cs = [1]
# # kernels = ['linear', 'rbf']
# kernels = ['linear']
# max_features_all = [None]
# # stop_words = [None]
#
#
# l=[]
# for kernel in kernels:
#     for C in Cs:
#         for max_features in max_features_all:
#             l.append([kernel, C, max_features])
#
#
# gs_numb = int(sys.argv[1])
# i = int(sys.argv[1])
# print(i)
# for parameters in l[gs_numb:gs_numb+6]:
#     # drop, batch_size, optimizer, activation2 = parameters
#
#     kernel, C, max_features = parameters
#     # if kernel == 'linear':
#     #     pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), min_df=2, max_features=max_features)),
#     #                          ('tfidf', TfidfTransformer()),
#     #                          ('clf', LogisticRegression(C=C, verbose=verbose, n_jobs=-1)), ])
#     # else:
#     #     pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), min_df=2, max_features=max_features)),
#     #                          ('tfidf', TfidfTransformer()),
#     #                          ('clf', SVC(C=C, kernel=kernel, verbose=verbose)), ])
#     clf = LogisticRegression(C=C, verbose=verbose, n_jobs=-1)
#     start = time.time()
#     clf.fit(Xtrain, Ytrain)
#     accuracy = clf.score(Xtest, Ytest)
#     with open(path_to_dir + 'log.txt', 'a+') as f:
#         end = time.time()
#         f.write(str(i)+'=================\n')
#         f.write('time: ' + str(np.round(end - start, 2))+'\n')
#         f.write('Parameters: '+str(parameters)+'\n')
#         f.write('Loss and Accuracy: ' +str(accuracy)+'\n')
#         f.write(str(np.round(accuracy,4)) + '\n\n')
#     i += 1
#


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



