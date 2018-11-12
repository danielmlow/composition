import numpy as np
import pandas as pd
import config
import data_helpers
import random
import datetime
import os
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score, f1_score
import plot_outputs

'''
obtain animate and inanimate sentences [sent1, sent2,] [Animate, INanimate, etc]
Then obtain the feature vector for each sentence. train and test split. and run through an SVM.
 
So:
1. obtain random sample from categories that seem to be animate and another sample from inanimate. 
1.1. Label each category as Anim. Inanimate. 
'''


categories = config.categories
Xtest, Ytest = data_helpers.load_data(config.test_path, categories)


category_animacy = pd.read_csv('animacy_labels.csv', header=None)
category_animacy.columns = ['category', 'animacy']
animate_categories = list(category_animacy.loc[category_animacy['animacy'] == 1].index)
inanimate_categories = list(category_animacy.loc[category_animacy['animacy'] == 0].index)


random.shuffle(animate_categories)
random.shuffle(inanimate_categories)

animate_categories = animate_categories[:24]
inanimate_categories = inanimate_categories[:24]

# Manually label sentences
# =================================================================================

# Obtain sentences of those categories

X_animate_to_choose_from = []
Y_animate_to_choose_from = []
for i in animate_categories:
    X_subset = Xtest[i*1200:i*1200+1200]
    Y_subset = list(range(i * 1200, i * 1200 + 1200))
    X_animate_to_choose_from.append(X_subset)
    Y_animate_to_choose_from.append(Y_subset)

X_animate_to_choose_from = [n for i in X_animate_to_choose_from for n in i]
Y_animate_to_choose_from = [n for i in Y_animate_to_choose_from for n in i]

X_inanimate_to_choose_from = []
Y_inanimate_to_choose_from = []
for i in inanimate_categories:
    X_subset = Xtest[i*1200:i*1200+1200]
    Y_subset = list(range(i * 1200, i * 1200 + 1200))
    X_inanimate_to_choose_from.append(X_subset)
    Y_inanimate_to_choose_from.append(Y_subset)

X_inanimate_to_choose_from = [n for i in X_inanimate_to_choose_from for n in i]
Y_inanimate_to_choose_from = [n for i in Y_inanimate_to_choose_from for n in i]

# Take random subsample
c = list(zip(X_animate_to_choose_from,Y_animate_to_choose_from))
random.shuffle(c)
X_animate_to_choose_from,Y_animate_to_choose_from = zip(*c)


# Take random subsample
c = list(zip(X_inanimate_to_choose_from,Y_inanimate_to_choose_from))
random.shuffle(c)
X_inanimate_to_choose_from,Y_inanimate_to_choose_from = zip(*c)

# # Manually filter them here:
dataset_length = 20000
X_inanimate_to_choose_from1 = X_inanimate_to_choose_from[:dataset_length ]
Y_inanimate_to_choose_from1 = Y_inanimate_to_choose_from[:dataset_length ]

X_animate_to_choose_from1 = X_animate_to_choose_from[:dataset_length ]
Y_animate_to_choose_from1 = Y_animate_to_choose_from[:dataset_length ]

#
# final_animate_sentences = []
# final_inanimate_sentences = []
#
# final_animate_ids = []
# final_inanimate_ids = []
#
# # Choose sentences
# for i,(y,x) in enumerate(zip(Y_animate_to_choose_from1,X_animate_to_choose_from1)):
#     print(str(i)+'===============')
#     print(x)
#     print('\n\n')
#     answer = input()
#     if answer=='a':
#         final_animate_sentences.append(x)
#         final_animate_ids.append(y)
#     elif answer=='n':
#         continue
#     elif answer=='q':
#         break
#


if config.local_or_cluster:
    # directory_name = 'svm_gs0_test'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = 'svm_animacy'
    input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/model_feature_vectors/'
else:
    # directory_name = 'svm_gs0_test'
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    file_name = os.path.basename(__file__)
    input_dir = '/model_feature_vectors/'

print('running '+directory_name+' '+file_name)

path_to_dir = os.path.join(config.save_to, directory_name + '/')
try: os.makedirs(path_to_dir)
except: pass
print('directory_name: '+directory_name)
print('path_to_dir: '+path_to_dir)





# SVM: classifying animacy
# =====================================================================================================================
stimuli  = np.concatenate([Y_animate_to_choose_from,Y_inanimate_to_choose_from])

np.save(path_to_dir+'stimuli_anim_inanimate',stimuli)

# LogReg w2v
w2v_prob = np.load(input_dir+'probability_output_w2v.npy')[stimuli]

w2v_coefficients = np.load(input_dir+'log_reg_coefficients_w2v.npy')
Xtest = np.load(config.word_embeddings_path+'Xtest_w2v_mean.npy')
Xtest = np.array([list(n) for n in Xtest])[stimuli]
raw_all = []
for mean_embedding in Xtest:
    raw = []
    for log_reg_boundary in w2v_coefficients:
        prob_class_i = np.dot(mean_embedding,log_reg_boundary)
        raw.append(prob_class_i)
    raw_all.append(raw)

w2v_raw = np.array(raw_all)

Xtest = np.load(config.word_embeddings_path+'Xtest_w2v_mean.npy')
w2v_mean = np.array([list(n) for n in Xtest])[stimuli]


# LogReg tfidf
tfidf_prob = np.load(input_dir+'probability_output_tfidf.npy')[stimuli]

# CNN softmax

# input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/model_feature_vectors/'
# np.savez_compressed(input_dir+'conv1', a = np.load(input_dir+'cnn_output_layers.npz')['a'])
# np.savez_compressed(input_dir+'conv2', a = np.load(input_dir+'cnn_output_layers.npz')['c'])
# np.savez_compressed(input_dir+'dense_final', a = np.load(input_dir+'cnn_output_layers.npz')['f'])

cnn_conv1 = np.load(input_dir+'conv1.npz')['a'][stimuli]
cnn_conv2 = np.load(input_dir+'conv2.npz')['a'][stimuli]
cnn_dense = np.load(input_dir+'cnn_dense_final.npz')['a'][stimuli]

# LSTM CNN softmax
lstm_dense = np.load(input_dir+'lstm_output_layers_dense_final.npz')['a'][stimuli]


# models =        [tfidf_prob, w2v_mean, w2v_raw, w2v_prob, cnn_conv1, cnn_conv2, cnn_dense, lstm_dense]
# model_names =   ['tfidf_prob','avg_w2v','w2v_raw','w2v_prob','cnn_conv1', 'cnn_conv2', 'cnn_dense_final', 'lstm_dense_final']

models =        [tfidf_prob, w2v_mean, w2v_raw, w2v_prob, cnn_dense, lstm_dense, cnn_conv1, cnn_conv2]
model_names =   ['tfidf_prob','avg_w2v','w2v_raw','w2v_prob', 'cnn_dense_final', 'lstm_dense_final','cnn_conv1', 'cnn_conv2']

y = np.concatenate([np.ones(dataset_length),np.zeros(dataset_length)])



train_index = np.concatenate([list(range(int(dataset_length *0.8))),list(range(dataset_length ,int(dataset_length+dataset_length *0.8)))])
test_index = np.concatenate([list(range(int(dataset_length *0.8), dataset_length )),list(range(int(dataset_length+dataset_length *0.8),int(dataset_length*2)))])


# SVM
# ===============================================================================================

start = time.time()
clf = LinearSVC()

with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write(file_name + '\n')
    f.write(directory_name + '\n\n')
    f.write(str(clf.get_params()) + '\n\n')

f1_scores = []
for model,model_name in zip(models, model_names):
    X_train = model[train_index]
    X_test = model[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)
    print('training model...'+str(model_name))
    clf.fit(X_train, y_train)
    y_predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predictions)
    f1  = f1_score(y_test, y_predictions)
    clas_rep = classification_report(y_test, y_predictions, target_names=['Inanimate', 'Animate'])
    df_clas_rep, df_clas_rep_latex = plot_outputs.classification_report_df(clas_rep)
    f1_scores.append(np.round(f1 * 100, 2))
    with open(path_to_dir + 'log.txt', 'a+') as f:
        end = time.time()
        f.write(str(model_name) + '=================\n')
        f.write('time: ' + str(np.round(end - start, 2)) + '\n')
        f.write('Accuracy: ' + str(np.round(accuracy * 100, 3)) + '\n')
        f.write('f1 score: ' + str(np.round(f1 * 100, 3)) + '\n')
        f.write('Classification Report: \n' + df_clas_rep_latex)
        f.write('\n\n')


with open(path_to_dir + 'log.txt', 'a+') as f:
    f.write('all f1 scores: '+str(f1_scores))
