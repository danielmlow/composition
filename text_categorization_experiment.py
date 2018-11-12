import numpy as np
import pandas as pd
import config
import data_helpers

categories = config.categories
Xtest, Ytest = data_helpers.load_data(config.test_path, categories)

input_dir = '/Users/danielmlow/Dropbox/cnn/mturk/categorization/results/'
df = pd.read_csv(input_dir+'results.csv')
ids = list(df.ids)

humans = np.array(df.humans)

dbpedia = [Ytest[n] for n in ids]

#  prediction for 64-way models
# =====================
# LogReg w2v
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_w2v_final/'
w2v = np.load(input_dir+'Ypredict_integer.npy')[ids]

# LogReg tfidf
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_tfidf_final/'
tfidf = np.load(input_dir+'Ypredict_integer.npy')[ids]

# CNN
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all/'
cnn = np.load(input_dir+'Ypredict_integer.npy')[-76800:][ids]

# LSTM
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/lstm18/'
lstm = np.load(input_dir+'Ypredict_integer.npy')[ids]

models = [tfidf,w2v, cnn, lstm]
human_scores = []
dbpedia_scores = []
for i in models:
    human_acc = np.round(np.sum(i==humans)/len(humans)*100,2)
    human_scores.append(human_acc)
    dbpedia_acc = np.round(np.sum(i == dbpedia) / len(dbpedia) * 100, 2)
    dbpedia_scores.append(dbpedia_acc)

#  prediction for 4-way models
# =====================
# transform original ids (range 76800) to range 4800.
categories = ['MusicGenre', 'Song', 'MusicalArtist', 'Album']
# if original id is between x and y then subtract x

ids_4 = []
for id in ids:
    if id>46800 and id<48000:
        new_id = id-46800
        ids_4.append(new_id)
    elif id>62400 and id<63600:
        new_id = id-62400+(1200*1)
        ids_4.append(new_id)
    elif id>45600 and id<46800:
        new_id = id-45600+(1200*2)
        ids_4.append(new_id)
    elif id>2400 and id<3600:
        new_id = id-2400+(1200*3)
        ids_4.append(new_id)


# Sanity check
categories_4 = ['MusicGenre', 'Song', 'MusicalArtist', 'Album']
Xtest_4, Ytest_4= data_helpers.load_data(config.test_path, categories_4)

for i in range(len(ids)):
    print(Xtest[ids[i]] + '\n')
    print(Xtest_4[ids_4[i]] + '\n')
    print('\n\n')

# Song: if original id is between x and y then subtract x +1200*1
#  MusicalArtist: if original id is between x and y then subtract x +1200*2




# Load
# ================================================================================
# LogReg w2v
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_w2v_4/'
w2v = np.load(input_dir+'Ypredict_integer.npy')[ids_4]

# LogReg tfidf
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_tfidf_4/'
tfidf = np.load(input_dir+'Ypredict_integer.npy')[ids_4]

# CNN
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_4/'
cnn = np.load(input_dir+'Ypredict_integer.npy')[-76800:][ids_4]

# LSTM
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/lstm18_4/'
lstm = np.load(input_dir+'Ypredict_integer.npy')[ids_4]


# transform humans to 4-way classification
categories_4 = ['MusicGenre', 'Song', 'MusicalArtist', 'Album']
humans_4 = []
for i in humans:
    if i == 52:
        humans_4.append(1)
    elif i == 39:
        humans_4.append(0)
    elif i == 2:
        humans_4.append(3)
    elif i == 38:
        humans_4.append(2)

dbpedia_4 = [Ytest_raw[n] for n in ids_4]




models = [tfidf,w2v, cnn, lstm]
human_scores = []
dbpedia_scores = []
for i in models:
    human_acc = np.round(np.sum(i==humans_4)/len(humans_4)*100,2)
    human_scores.append(human_acc)
    dbpedia_acc = np.round(np.sum(i == dbpedia_4) / len(dbpedia_4) * 100, 2)
    dbpedia_scores.append(dbpedia_acc)

for i in dbpedia_scores:
    print(i)


# Entropy per response on first 4 subjects
from scipy.stats import entropy

a = entropy([0,0.33,0.33,0.33])
print(a)

