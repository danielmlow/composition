from rouge import Rouge
import itertools
import data_helpers
import config
import pandas as pd
import numpy as np


# Rouge metric
# ============================================================================

categories = config.categories

X, y= data_helpers.load_whole_dataset(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)
rouge = Rouge()

# d2c
c2d = pd.read_csv('/Users/danielmlow/Dropbox/cnn/experiment/final_sentences_experiment/other/'+'c2d_cleaned.csv')
d2c = pd.read_csv('/Users/danielmlow/Dropbox/cnn/experiment/final_sentences_experiment/other/'+'d2c_cleaned.csv')


ids= list(d2c['ids'])
sentences = np.array(X)[ids]
d = {}
for i in range(0,len(sentences),6):
    scores_one_category = []
    sentences_one_category = sentences[i:i+6]
    combinations = itertools.combinations(sentences_one_category, 2)
    for j in combinations:
        hypothesis = j[0]
        reference = j[1]
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-1').get('f')
        scores_one_category.append(one_f)
    mean_score = np.mean(scores_one_category)
    d[i]=mean_score.round(4)

d2c_rouge_mean = np.mean(list(d.values()))


# c2d
ids= list(c2d['ids'])
sentences = np.array(X)[ids]
d = {}

for i in range(0,len(sentences),6):
    scores_one_category = []
    scores_2 = []
    sentences_one_category = sentences[i:i+6]
    combinations = itertools.combinations(sentences_one_category, 2)
    for j in combinations:
        hypothesis = j[0]
        reference = j[1]
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-1').get('f')
        # score_2 = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        scores_one_category.append(one_f)
        # scores_2.append(score_2)
    d[i]=np.mean(scores_one_category).round(4)

c2d_rouge_mean = np.mean(list(d.values()))



# Now do for each cluster
# ==========================================================================================================================================================================================

X_array = np.array(X)

d0 = {}
rouge_scores = []
for i in range(64):
    category = categories[i]
    ids= list(d2c[d2c.category_name== category].ids)
    sentences_one_category = X_array[ids]
    scores_one_category = []
    scores_2 = []
    combinations = itertools.combinations(sentences_one_category, 2)
    for j in combinations:
        hypothesis = j[0]
        reference = j[1]
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        # score_2 = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        scores_one_category.append(one_f)
        # scores_2.append(score_2)
    d0[i] = np.mean(scores_one_category).round(4)



d1={}
rouge_scores = []
for i in range(64):
    category = categories[i]
    ids= list(c2d[c2d.category_name== category].ids)
    sentences_one_category = X_array[ids]
    scores_one_category = []
    scores_2 = []
    combinations = itertools.combinations(sentences_one_category, 2)
    for j in combinations:
        hypothesis = j[0]
        reference = j[1]
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        # score_2 = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        scores_one_category.append(one_f)
        # scores_2.append(score_2)
    d1[i] = np.mean(scores_one_category).round(4)


d2c_rouge_mean = np.nanmean(list(d0.values()))
c2d_rouge_mean = np.nanmean(list(d1.values()))









