

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import pearsonr, spearmanr

def cluster(n_clusters, layer, output_dir, file_name):
    clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(np.array(layer))
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # silhouette_avg = silhouette_score(layer, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.array(layer), cluster_labels)
    # silh_values.append(sample_silhouette_values)
    category = [[n]*332 for n in range(64)]
    category = [n for i in category for n in i]
    cluster_df = pd.DataFrame({'ids':ids, 'category':category, 'cluster_labels':cluster_labels, 'silh_values':sample_silhouette_values})
    cluster_df.to_csv(output_dir+'cluster_df_'+file_name+'.csv')
    return cluster_labels, sample_silhouette_values

output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/'
dense_final_df = pd.read_csv(output_dir+'whole_dataset/dense_final_filtered_vectors.csv', index_col=0)
conv_1_df = pd.read_csv(output_dir+'whole_dataset/conv_1_filtered_vectors.csv', index_col=0).iloc[:,1792]
ids = np.array(dense_final_df.index)



# Scotts method to clustering
# ==================================================================================================================
'''
Take 20 sentences at a 

'''

# create dictionary {cluster_0: [id_highest_silh_val,..., lowest_silh],
def dict_silh(cluster_labels, sample_silhouette_values):
    d_silh = {}
    d_ids = {}
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        id = ids[i]
        silh_value = sample_silhouette_values[i]
        d_silh.setdefault(cluster_label, []).append(silh_value)
        d_ids.setdefault(cluster_label, []).append(id)
    return d_silh, d_ids

# Rank slih scores and their ids. Take 24 highest.
def rank_silh(d_silh, d_ids):
    subset_number_start = 0
    subset_number = 20
    d_highest_silh = {}
    d_highest_ids = {}
    l=[]
    for i in range(64):
        cluster_silh_scores = d_silh.get(i)
        silh_scores_argsort = np.array(cluster_silh_scores).argsort()[::-1]
        cluster_ids = d_ids.get(i)
        ids_sorted = [ cluster_ids[j] for j in silh_scores_argsort] #this reorders list of sentence_ids from high to low
        cluster_silh_scores_sorted = cluster_silh_scores[:]
        cluster_silh_scores_sorted.sort(reverse=True)
        l.append((cluster_silh_scores_sorted[subset_number_start:subset_number ][-1])) #keep track of lowest values
        d_highest_silh[i]=np.round(cluster_silh_scores_sorted[subset_number_start:subset_number ], 4)
        d_highest_ids[i]=ids_sorted[subset_number_start:subset_number ]
    return d_highest_silh, d_highest_ids

n_clusters = 64

file_name = 'd2c'
cluster_labels, sample_silhouette_values = cluster(n_clusters, dense_final_df, output_dir, file_name)
# #
# np.save(output_dir+'cluster_labels_'+file_name,cluster_labels)
# np.save(output_dir+'sample_silhouette_values_'+file_name, sample_silhouette_values)

cluster_labels = np.load(output_dir+'cluster_labels_'+file_name+'.npy')
sample_silhouette_values = np.load(output_dir+'sample_silhouette_values_'+file_name+'.npy')
# create dictionary {cluster_0: [silh_value_id0, ...],....}


d_silh, d_ids = dict_silh(cluster_labels, sample_silhouette_values)
d_highest_silh, d_highest_ids = rank_silh(d_silh, d_ids)

# Check if correct. Compare score of highest id with highest silh score
# Get a sentence
# sent = ids_sorted[20]
# # Find index in silh_scores
# index_sent = cluster_ids.index(sent)
# # Get it's silh_score and compare to nth score
# print(cluster_silh_scores[index_sent]==cluster_silh_scores_sorted [20])


df_subset_silh = pd.DataFrame(d_highest_silh)
df_subset_ids = pd.DataFrame(d_highest_ids)

ids2 = df_subset_ids.values.T.flatten()
silh2 = df_subset_silh.values.T.flatten()

cluster_df_20 = pd.DataFrame({'ids':ids2, 'silh_values':silh2})

# Save results
cluster_df_20.to_csv(output_dir+'subset20_d2c.csv')



# TODO: check if mapping is correct between silh and id

df_subset_ids_flatten = df_subset_ids.values.T.flatten()
layer1 = conv_1_df.loc[df_subset_ids_flatten ].iloc[:,:1792]
layer3 = dense_final_df.loc[df_subset_ids_flatten ]


'''
Ok. So, after choosing 20 sentences with high silhouette value, we want the 6 that A) correlate the least among each other in layer1 (find lowest values in RSM of layer1) or B) the ones that correlate least with layer1 (take RSM of layer3 minus RSM of layer1 = differences between correlations at both layers, and then take highest values in resulting matrix).
'''

# Choose sentences that corr low on l1. Scotts method
# ===================================================================================================
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# layer = layer1.iloc[:, :1792]
method = 'pearson'

# RSM of all sentences (20*64)
corr_layerB = layer1.T.corr()

# Every 6 sentences in cluster, mean(6*6 sentences)-mean(6*1280). Remove highest.


final_sentences = []
final_sentences_values = []

for i in range(64):
    mean_within_all = []
    for j in range(20):
        mean_within = np.mean(corr_layerB.iloc[i * 20 + j,i * 20:i * 20 + 20 ])
        # mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :])  # mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :]).mean()
        elements_across = list(corr_layerB.iloc[i * 20 + j,:])
        del elements_across[i * 20:i * 20 + 20]
        mean_across = np.mean(elements_across)#mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :]).mean()
        difference = mean_within-mean_across
        mean_within_all.append(difference)
    smallest = np.argsort(mean_within_all)[:6] #TODO: maybe here also iteratively find the best 6 combination
    values = np.array(mean_within_all)[smallest]
    sentences = [i*20+n for n in smallest]
    indexes = corr_layerB.index[sentences]
    final_sentences.append(list(indexes))
    final_sentences_values.append(list(values))

final_sentences_d2c = [n for i in final_sentences for n in i]
np.save(output_dir+'final_ids_d2c.npy', final_sentences_d2c)

# c2d
# ========================================================================================================================
n_clusters = 64

file_name = 'c2d'
cluster_labels, sample_silhouette_values = cluster(n_clusters,conv_1_df.iloc[:,:1792], output_dir, file_name)
#
np.save(output_dir+'cluster_labels_'+file_name,cluster_labels)
np.save(output_dir+'sample_silhouette_values_'+file_name, sample_silhouette_values)

# cluster_labels = np.load(output_dir+'cluster_labels_c2d.npy')
# sample_silhouette_values = np.load(output_dir+'sample_silhouette_values_c2d.npy')

# create dictionary {cluster_0: [silh_value_id0, ...],...
d_silh, d_ids = dict_silh(cluster_labels, sample_silhouette_values)
d_highest_silh, d_highest_ids = rank_silh(d_silh, d_ids)

for i in d_ids.keys():
    if len(d_ids.get(i))<20:
        print(i,len(d_ids.get(i)))



df_subset_silh = pd.DataFrame(d_highest_silh)
df_subset_ids = pd.DataFrame(d_highest_ids)
ids2 = df_subset_ids.values.T.flatten()
silh2 = df_subset_silh.values.T.flatten()

cluster_df_20 = pd.DataFrame({'ids':ids2, 'silh_values':silh2})
cluster_df_20.to_csv(output_dir+'subset20_'+file_name+'.csv')

df_subset_ids_flatten = df_subset_ids.values.T.flatten()
layer1 = conv_1_df.loc[df_subset_ids_flatten ].iloc[:,:1792]
layer3 = dense_final_df.loc[df_subset_ids_flatten ]

# Correlating least on df
# ================================================================================================
# layer = dense_final_df.copy()
method = 'pearson'

# RSM of all sentences (20*64)
corr_layerB =layer3.T.corr()
# Every 6 sentences in cluster, mean(6*6 sentences)-mean(6*1280). Remove highest.


final_sentences = []
final_sentences_values = []

for i in range(64):
    mean_within_all = []
    for j in range(20): #TODO: add combinations of sentences in 20
        mean_within = np.mean(corr_layerB.iloc[i * 20 + j,i * 20:i * 20 + 20 ])
        # mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :])  # mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :]).mean()
        elements_across = list(corr_layerB.iloc[i * 20 + j,:])
        del elements_across[i * 20:i * 20 + 20]
        mean_across = np.mean(elements_across)#mean_across = np.mean(corr_layerB.iloc[i * 20 + j, :]).mean()
        difference = mean_within-mean_across
        mean_within_all.append(difference)
    smallest = np.argsort(mean_within_all)[:6] #TODO: maybe here also iteratively find the best 6 combination
    values = np.array(mean_within_all)[smallest]
    sentences = [i*20+n for n in smallest]
    indexes = corr_layerB.index[sentences]
    final_sentences.append(list(indexes))
    final_sentences_values.append(list(values))


final_sentences_c2d = [n for i in final_sentences for n in i]

# corr_layer1_all = layer1.loc[final_sentences_c2d].T.corr(method=method)
# corr_layer3_all = layer3.loc[final_sentences_c2d].T.corr(method=method)
# corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
# corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
# corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()
#
# print(corr_between_layers.round(4))
# # 0.5683
np.save(output_dir+'final_ids_c2d.npy', final_sentences_c2d)

# Correlations
# ==============================================================================================================================
def triu(corr_conv_1_d2c):
    corr_conv_1_d2c_triu = corr_conv_1_d2c.where(np.triu(np.ones(corr_conv_1_d2c.shape)).astype(np.bool)).reset_index(drop=True)
    return corr_conv_1_d2c_triu

# Replace diagonal
def remove_diagonal(conv1_corr):
    for i in range(conv1_corr.shape[0]):
            conv1_corr.iloc[i,i] = np.nan
    return conv1_corr

def remove_nan(corr_df_c2d_triu):
    corr_df_c2d_triu = list(corr_df_c2d_triu[np.logical_not(np.isnan(corr_df_c2d_triu))])
    return corr_df_c2d_triu



final_ids_c2d = np.load(output_dir+'final_ids_c2d.npy')
corr_conv_1_c2d = conv_1_df.loc[final_ids_c2d ].iloc[:, :1792].T.corr()
corr_conv_1_c2d_triu = triu(corr_conv_1_c2d)
corr_conv_1_c2d_triu = remove_diagonal(corr_conv_1_c2d_triu)
corr_conv_1_c2d_triu = corr_conv_1_c2d_triu.values.flatten()
corr_conv_1_c2d_triu = remove_nan(corr_conv_1_c2d_triu)

corr_df_c2d = dense_final_df.loc[final_ids_c2d ].T.corr()
corr_df_c2d_triu = triu(corr_df_c2d)
corr_df_c2d_triu = remove_diagonal(corr_df_c2d_triu)
corr_df_c2d_triu = corr_df_c2d_triu.values.flatten()
corr_df_c2d_triu = remove_nan(corr_df_c2d_triu)



final_ids_d2c = np.load(output_dir+'final_ids_d2c.npy')
corr_conv_1_d2c = conv_1_df.loc[final_ids_d2c].iloc[:, :1792].T.corr()
corr_conv_1_d2c_triu = triu(corr_conv_1_d2c)
corr_conv_1_d2c_triu = remove_diagonal(corr_conv_1_d2c_triu)
corr_conv_1_d2c_triu = corr_conv_1_d2c_triu.values.flatten()
corr_conv_1_d2c_triu = remove_nan(corr_conv_1_d2c_triu)

corr_df_d2c = dense_final_df.loc[final_ids_d2c].T.corr()
corr_df_d2c_triu = triu(corr_df_d2c)
corr_df_d2c_triu = remove_diagonal(corr_df_d2c_triu)
corr_df_d2c_triu = corr_df_d2c_triu.values.flatten()
corr_df_d2c_triu = remove_nan(corr_df_d2c_triu)


# Combine
corr_conv_1_all = conv_1_df.loc[np.concatenate((final_ids_c2d, final_ids_d2c))].iloc[:, :1792].T.corr()
corr_conv_1_all_triu = triu(corr_conv_1_all)
corr_conv_1_all_triu = remove_diagonal(corr_conv_1_all_triu)
corr_conv_1_all_triu = corr_conv_1_all_triu.values.flatten()
corr_conv_1_all_triu = remove_nan(corr_conv_1_all_triu)

corr_df_all = dense_final_df.loc[np.concatenate((final_ids_c2d, final_ids_d2c))].T.corr()
corr_df_all_triu = triu(corr_df_all)
corr_df_all_triu = remove_diagonal(corr_df_all_triu)
corr_df_all_triu = corr_df_all_triu.values.flatten()
corr_df_all_triu = remove_nan(corr_df_all_triu)

print('1. correlation between triu of conv1 from c2d and DF from c2d: ', pearsonr(corr_conv_1_c2d_triu,corr_df_c2d_triu  )[0].round(2))
print('2. correlation between triu of conv1 from d2c and DF from d2c: ', pearsonr(corr_conv_1_d2c_triu, corr_df_d2c_triu )[0].round(2))
print('3. correlation between triu of conv1 from selected and DF from selected: ', pearsonr(corr_conv_1_all_triu, corr_df_all_triu )[0].round(2))

# print('1. correlation between triu of conv1 from c2d and DF from c2d: ', 0.568)
# print('2. correlation between triu of conv1 from d2c and DF from d2c: ', 0.421)
# print('3. correlation between triu of conv1 from c2d and DF from d2c: ', 0.521)



# Summerize
# ============================================================
final_ids_d2c = np.load(output_dir+'final_ids_d2c.npy')
final_ids_c2d = np.load(output_dir+'final_ids_c2d.npy')

# sents_d2c = []
# for i in range(0, len(final_ids_d2c[:]), 6):
#     sents_d2c.append([i:i+2)
#
import config
categories = config.categories
#
category_list = [[n]*6000 for n in categories]
category_list = np.array(category_list).flatten()

cluster_results_c2d = pd.read_csv(output_dir+'subset20_c2d.csv')
cluster_results_d2c = pd.read_csv(output_dir+'subset20_d2c.csv')

ids_20_c2d = cluster_results_c2d['ids']
ids_20_d2c = cluster_results_d2c['ids']

silh_20_c2d = cluster_results_c2d['silh_values']
silh_20_d2c = cluster_results_d2c['silh_values']

cluster_number = np.array([[n]*6 for n in range(64)]).flatten()

import data_helpers
X_raw, y= data_helpers.load_whole_dataset_raw(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)

summary_d2c = pd.DataFrame()
summary_d2c['id']=final_ids_d2c
summary_d2c['sentence']=np.array(X_raw)[final_ids_d2c]
summary_d2c['category']=category_list[final_ids_d2c]
summary_d2c['cluster']= cluster_number
summary_d2c['silh_value']= silh_20_d2c
summary_d2c = summary_d2c[['id', 'sentence','category', 'cluster', 'silh_value']]
summary_d2c.to_csv(output_dir+'summary_d2c.csv',index=False, encoding='utf-8-sig')

summary_c2d = pd.DataFrame()
summary_c2d['id']=final_ids_c2d
summary_c2d['sentence']=np.array(X_raw)[final_ids_c2d]
summary_c2d['category']=category_list[final_ids_c2d]
summary_c2d['cluster']= cluster_number
summary_c2d['silh_value']= silh_20_c2d
summary_c2d = summary_c2d[['id', 'sentence','category', 'cluster', 'silh_value']]
summary_c2d.to_csv(output_dir+'summary_c2d.csv',index=False, encoding='utf-8-sig')


# repeated sentences
counter = 0
for i in ids_c:
    if i in ids_d:
        counter+=1



# Highest silh
# ========================================================================================================================
df = pd.read_csv(output_dir+'subset20_c2d.csv')

sentences_c2d_20 = np.array(X_raw)[df['ids']]
scores_c2d_20 = df['ids'][df['silh_values']]

pd.DataFrame(sentences_c2d_20).to_csv(output_dir+'sentences_c2d_20.csv', index=False, encoding='utf-8-sig')

# Recluster
# ========================================================================================================================
# 1. d2c
# Load 20 per cluster
output_dir2 = output_dir+'recluster/'
top_20 = np.array(pd.read_csv(output_dir+'subset20_d2c.csv')['ids'])
dense_final_df



# Compare to lstm
# ========================================================================================================================
lstm_dense = pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/lstm17_final_1layer/output_layers_whole_dataset_d.npz')['a'])


corr_lstm_all = lstm_dense.loc[final_ids_d2c].T.corr()
corr_lstm_all_triu = triu(corr_lstm_all)
corr_lstm_all_triu = remove_diagonal(corr_lstm_all_triu)
corr_lstm_all_triu = corr_lstm_all_triu.values.flatten()
corr_lstm_all_triu = remove_nan(corr_lstm_all_triu)


corr_df_all = dense_final_df.loc[final_ids_d2c].T.corr()
corr_df_all_triu = triu(corr_df_all)
corr_df_all_triu = remove_diagonal(corr_df_all_triu)
corr_df_all_triu = corr_df_all_triu.values.flatten()
corr_df_all_triu = remove_nan(corr_df_all_triu)


print('3. correlation between triu of conv1 from selected and DF from selected: ', spearmanr(corr_lstm_all_triu , corr_df_all_triu )[0].round(2))


# Scotts
# ========================================================================================================================
scott_ids = sio.loadmat(output_dir+'forDLv3.mat')
scott_ids.keys()

selected = scott_ids.get('selected')[0]
final_sentences =[n-1 for n in selected]

d2c = scott_ids.get('d2c')[0]
c2d = scott_ids.get('c2d')[0]

final_sentences_d2c =[n-1 for n in d2c]
final_sentences_c2d =[n-1 for n in c2d]


final_sentences = np.random.randint(0,21000,768)


corr_conv_1_all = conv_1_df.iloc[np.concatenate((final_sentences_d2c, final_sentences_c2d))].iloc[:, :1792].T.corr()
corr_conv_1_all = conv_1_df.iloc[final_sentences].iloc[:, :1792].T.corr()
corr_conv_1_all_triu = triu(corr_conv_1_all)
corr_conv_1_all_triu = remove_diagonal(corr_conv_1_all_triu)
corr_conv_1_all_triu = corr_conv_1_all_triu.values.flatten()
corr_conv_1_all_triu = remove_nan(corr_conv_1_all_triu)


corr_df_all = dense_final_df.iloc[np.concatenate((final_sentences_d2c, final_sentences_c2d))].T.corr()
corr_df_all = dense_final_df.iloc[final_sentences].T.corr()

corr_df_all_triu = triu(corr_df_all)
corr_df_all_triu = remove_diagonal(corr_df_all_triu)
corr_df_all_triu = corr_df_all_triu.values.flatten()
corr_df_all_triu = remove_nan(corr_df_all_triu)


print('3. correlation between triu of conv1 from selected and DF from selected: ', pearsonr(corr_conv_1_all_triu, corr_df_all_triu )[0].round(2))

# Rouge metric
# ============================================================================
from rouge import Rouge
import itertools

X, y= data_helpers.load_whole_dataset(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)
rouge = Rouge()

# d2c
ids= list(summary_d2c['id'])
sentences = np.array(X)[ids]
d = {}
for i in range(0,len(sentences),6):
    scores_one_category = []
    sentences_one_category = sentences[i:i+6]
    combinations = itertools.combinations(sentences_one_category, 2)
    for j in combinations:
        hypothesis = j[0]
        reference = j[1]
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        scores_one_category.append(one_f)
    mean_score = np.mean(scores_one_category)
    d[i]=mean_score.round(4)

d2c_rouge_mean = np.mean(list(d.values()))


# c2d
ids= list(summary_c2d['id'])
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
        one_f = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        # score_2 = rouge.get_scores(hypothesis, reference)[0].get('rouge-2').get('f')
        scores_one_category.append(one_f)
        # scores_2.append(score_2)
    d[i]=np.mean(scores_one_category).round(4)

c2d_rouge_mean = np.mean(list(d.values()))





