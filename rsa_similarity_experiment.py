import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance
from matplotlib import rcParams
import config
import data_helpers
from sklearn.preprocessing import StandardScaler

from numpy.random import seed
seed(123)

from scipy.stats.mstats import spearmanr
from scipy.stats.mstats import pearsonr

def extract_triu(rsm):
    triu = rsm.where(np.triu(np.ones(rsm.shape)).astype(np.bool)).reset_index(drop=True)
    return triu

# Replace diagonal
def remove_diagonal(triu):
    for i in range(triu.shape[0]):
            triu.iloc[i,i] = np.nan
    return triu

def remove_nan(triu):
    triu = list(triu[np.logical_not(np.isnan(triu))])
    return triu


def layer_to_flattened_triu(layer):
    rsm = layer.T.corr(method=method)
    triu = extract_triu(rsm)
    triu = remove_diagonal(triu)
    triu = triu.values.flatten()
    triu = remove_nan(triu)
    return triu

def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    standardized_data = scaler.transform(data)
    return standardized_data

def normalize(vector):
    mean = np.mean(vector)
    vector_normalized = [n-mean for n in vector]
    return vector_normalized

def plot_cluster_map(output_dir, df, column_names, output_file_name='lstm_1'):
    # layer_name is just used to define the output layer name
    df.columns = column_names
    df.index = column_names
    sns.set(font_scale=1)
    cg = sns.heatmap(df, method='ward',cmap="RdBu_r", vmin = -1., vmax=1.0, cbar_kws={"ticks":[-1., -0.5, 0.0, 0.5, 1.0]})
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    cg.savefig(output_dir + 'RSA_ward_'+ output_file_name+ '.eps', format='eps', dpi=100)


def plot_heatmap(output_dir , df_corr, column_names,output_file_name = 'similarity_experiment', with_corr_values=True):
    # layer_name is just used to define the output layer name
    plt.clf()
    df_corr.columns = column_names
    df_corr.index = column_names
    sns.set(font_scale=1.8)
    sns.heatmap(df_corr,cmap="RdBu_r", vmin = -1., vmax=1.0, cbar_kws={"ticks":[-1., -0.5, 0.0, 0.5, 1.0]}, annot=with_corr_values)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout(1.8)
    plt.savefig(output_dir + 'RSA_ward_'+ output_file_name+ '.eps', format='eps', dpi=100)


def add_diagonal(df_corr):
    for i in range(df_corr.shape[0]):
            df_corr.iloc[i,i] = 6
    return df_corr


def subplot_heatmap(df_corr, add_x_labels, add_y_labels, column_names ):
    empty = ['' for n in range(18)]
    if add_x_labels:
        df_corr.columns = column_names
    else:
        df_corr.columns = empty
    if add_y_labels:
        df_corr.index = column_names
    else:
        df_corr.index= empty
    sns.set(font_scale=1)
    sns.heatmap(df_corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0,  cbar=False, annot=False)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)


# Import sentences
# ==========
categories = config.categories

Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=False) # I changed this so it combines train and test
Xtest, Ytest = data_helpers.load_data(config.test_path, categories)
Xtest_raw, Ytest_raw = data_helpers.load_data_raw(config.test_path, categories)

# Import results
# ===========================================================================

input_dir = '/Users/danielmlow/Dropbox/cnn/mturk/semantic_similarity/results/'
df = pd.read_csv(input_dir+'aggregated.csv')
id1 = list(df['id1'])
id2 = list(df['id2'])
humans = list(df['result'])
stimuli = list(set(np.concatenate([id1, id2])))


# RSM of RSMs with a random subset from test set
# ===========================================================================
# LogReg w2v

stimuli=range(1,76800,50)

input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/model_feature_vectors/'
output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'
w2v = pd.DataFrame(np.load(input_dir+'probability_output_w2v.npy')[stimuli], index=stimuli)
# LogReg tfidf
tfidf = pd.DataFrame(np.load(input_dir+'probability_output_tfidf.npy')[stimuli], index=stimuli)
# tfidf = pd.DataFrame(np.loadtxt(input_dir+'probability_output.txt', delimiter=' ', dtype = str)[stimuli], index=stimuli)

# CNN dense
cnn_dense = pd.DataFrame(np.load(input_dir+'cnn_dense_final.npz')['a'][stimuli], index=stimuli)

# LSTM CNN dense
lstm_dense = pd.DataFrame(np.load(input_dir+'lstm_output_layers_dense_final.npz')['a'][stimuli], index=stimuli)


# RSM of RSMs
method = 'pearson'
w2v_triu = layer_to_flattened_triu(w2v)
tfidf_triu = layer_to_flattened_triu(tfidf)
cnn_triu = layer_to_flattened_triu(cnn_dense)
# conv1_triu = (cnn.iloc[:, :1792])
lstm_triu = layer_to_flattened_triu(lstm_dense)


column_names = ['w2v', 'Tfidf', 'CNN', 'LSTM']
df_norm = pd.DataFrame([normalize(w2v_triu),normalize(tfidf_triu), normalize(cnn_triu), normalize(lstm_triu)])
method='pearson'
df_norm_corr = df_norm.T.corr(method=method)
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'rsm_of_rsms_test_set2_'+method, with_corr_values=True)

method='spearman'
df_norm_corr = df_norm.T.corr(method=method)
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'rsm_of_rsms_test_set2_'+method, with_corr_values=True)









# Import layers for similarity experiment
# ===========================================================================

stimuli  = np.sort(stimuli)
# LogReg w2v
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_w2v_final/'
w2v = pd.DataFrame(np.load(input_dir+'probability_output.npy')[stimuli], index=stimuli)

w2v_6 = pd.DataFrame(np.load(input_dir+'probability_output.npy')[stimuli], index=stimuli)

w2v_coefficients = np.load(input_dir+'log_reg_coefficients.npy')
Xtest = np.load(config.word_embeddings_path+'Xtest_w2v_mean.npy')
Xtest = np.array([list(n) for n in Xtest])[stimuli]
raw_all = []
for mean_embedding in Xtest:
    raw = []
    for log_reg_boundary in w2v_coefficients:
        prob_class_i = np.dot(mean_embedding,log_reg_boundary)
        raw.append(prob_class_i)
    raw.append(raw)

w2v_all = pd.DataFrame(raw_all, index=stimuli)


# LogReg tfidf
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_tfidf_final/'
tfidf = pd.DataFrame(np.load(input_dir+'probability_output.npy')[stimuli], index=stimuli)
# tfidf = pd.DataFrame(np.loadtxt(input_dir+'probability_output.txt', delimiter=' ', dtype = str)[stimuli], index=stimuli)

# CNN softmax
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all/'
cnn = pd.DataFrame(np.load(input_dir+'output_layers_whole_dataset_g.npz')['a'][-76800:][stimuli], index=stimuli)
cnn_dense = pd.DataFrame(np.load(input_dir+'output_layers_whole_dataset_f.npz')['a'][-76800:][stimuli], index=stimuli)

# LSTM CNN softmax
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/lstm18/'
lstm= pd.DataFrame(np.load(input_dir+'output_layers_softmax_final.npz')['a'][stimuli], index=stimuli)
lstm_dense = pd.DataFrame(np.load(input_dir+'output_layers_dense_final.npz')['a'][stimuli], index=stimuli)

# HUMAN RSM
human_rsm = pd.DataFrame(np.zeros([18,18]), index=stimuli, columns=stimuli)
d = {}
for i in range(len(humans)):
    sentence1 = id1[i]
    sentence2 = id2[i]
    score = humans[i]
    d[sentence1,sentence2]=score
    human_rsm.loc[sentence1, sentence2]=score
    human_rsm.loc[sentence2, sentence1] = score

df.sort_index(ascending=False)
human_rsm = human_rsm[np.sort(stimuli)]
human_rsm = human_rsm.loc[np.sort(stimuli),:]


human_triu = extract_triu(human_rsm)
human_triu = remove_diagonal(human_triu)
human_triu = human_triu.values.flatten()
human_triu = remove_nan(human_triu)






# # Other feature vectors
# # LogReg w2v
# input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_w2v_final/'
# w2v_coefficients = np.load(input_dir+'log_reg_coefficients.npy')
# Xtest = np.load(config.word_embeddings_path+'Xtest_w2v_mean.npy')
# Xtest = np.array([list(n) for n in Xtest])[stimuli]
# softmax_all = []
# for mean_embedding in Xtest:
#     softmax = []
#     for log_reg_boundary in w2v_coefficients:
#         prob_class_i = np.dot(mean_embedding,log_reg_boundary)
#         softmax.append(prob_class_i)
#     softmax_all.append(softmax)
#
# w2v = pd.DataFrame(softmax_all, index=stimuli)
#
# # Sanity Check (change i in both lines)
# predicted_category = np.argmax(w2v.iloc[2])
# categories[predicted_category]
# # the above category should be the category of:
# Xtest_raw[stimuli[2]]

# RSM of RSMs
# ===============================================================================================


column_names = list(range(1,19))

left  = 0.5  # the left side of the subplots of the figure
right = 1    # the right side of the subplots of the figure
bottom = 1   # the bottom of the subplots of the figure
top = 1      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
plt.subplots_adjust(right=right,top=top)


# rsms = [w2v.T.corr(), tfidf.T.corr(),cnn.T.corr(),lstm.T.corr()]
# fig, axes = plt.subplots(nrows=2, ncols=2)
# sns.set(font_scale=1)
# for i,ax in enumerate(axes.flat):
#     print(i)
#     print(ax)
#     print('====')
#     im = ax.sns.heatmap(df_corr, cmap="RdBu_r", vmin=-1., vmax=1.0,  cbar=False,
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# plt.show()

method = 'spearman'
column_names =columns
#  column_names = list(range(1,19))
plt.clf()
plt.figure(1, figsize=(10,10))
plt.subplot(221)
w2v_corr = pd.DataFrame(normalize(w2v.T.corr(method=method).values))
subplot_heatmap(w2v_corr, False, True, column_names)
plt.title('Averaged word embeddings')
plt.ylabel([])
plt.subplot(222)
tfidf_corr = pd.DataFrame(normalize(tfidf.T.corr(method=method).values))
subplot_heatmap(tfidf_corr, False, False, column_names)
plt.title('Tf-idf')
plt.subplot(223)
cnn_corr = pd.DataFrame(normalize(cnn_dense.T.corr(method=method).values))
subplot_heatmap(cnn_corr, True, True, column_names)
plt.title('CNN')
plt.subplot(224)
lstm_corr = pd.DataFrame(normalize(lstm_dense.T.corr(method=method).values))
subplot_heatmap(lstm_corr , True, False, column_names)
plt.title('LSTM')
# plt.subplot(313, sharex=ax1, sharey=ax1)
plt.tight_layout(1.8)
plt.savefig(output_dir+'rsm_of_models_sorted'+ '.eps', format='eps', dpi=100)

# Plot human rsm
human_rsm = add_diagonal(human_rsm)
human_rsm_flattened = human_rsm.values.flatten()
human_rsm_scaled = pd.DataFrame(np.reshape(np.interp(human_rsm_flattened , (human_rsm_flattened .min(), human_rsm_flattened.max()), (-1, +1)), (18,18)))

categories = ['Decoration', 'MilitaryConflict', 'MilitaryPerson', 'Monarch','Politician', 'University']
columns = [[n]*3 for n in categories]
columns = [n for i in columns for n in i]

human_rsm_scaled = normalize(human_rsm_scaled.values)
human_rsm_scaled = pd.DataFrame(human_rsm_scaled)
human_rsm_scaled.columns = columns
human_rsm_scaled.index = columns


plot_heatmap(output_dir,human_rsm_scaled, columns, 'human_rsm_sorted',False)

human_triu = extract_triu(human_rsm_scaled)
human_triu = remove_diagonal(human_triu)
human_triu = human_triu.values.flatten()
human_triu = remove_nan(human_triu)





# RSM of RSMs
# ===============================================================================================
w2v_triu = layer_to_flattened_triu(w2v)
tfidf_triu = layer_to_flattened_triu(tfidf)
cnn_triu = layer_to_flattened_triu(cnn)
# conv1_triu = (cnn.iloc[:, :1792])
lstm_triu = layer_to_flattened_triu(lstm)

# df = pd.DataFrame([w2v_triu,tfidf_triu, cnn_triu, lstm_triu, human_triu])
# df_corr = df.T.corr()
# column_names = ['w2v', 'Tfidf', 'CNN', 'LSTM', 'Human']
# output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'
# plot_cluster_map(output_dir, df_corr, column_names,output_file_name = 'similarity_experiment_spearman')
#
# plot_heatmap(output_dir, df_corr, column_names,output_file_name = 'similarity_experiment_heatmap_spearman', with_corr_values=True)
#
# column_names = ['w2v', 'Tfidf', 'CNN', 'LSTM', 'Human']
# # df_norm = pd.DataFrame([normalize(w2v_triu),normalize(tfidf_triu), normalize(cnn_triu), normalize(lstm_triu), normalize(human_triu)])
# df_norm = pd.DataFrame([w2v_triu,normalize(tfidf_triu), normalize(cnn_triu), normalize(lstm_triu), normalize(human_triu)])
# df_norm_corr = df_norm.T.corr(method='')
# plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'similarity_experiment_heatmap_normalized_pearson', with_corr_values=True)
#

# with dense_layer
# ============================================================================================
# w2v_triu_raw = layer_to_flattened_triu(w2v_raw) #don't think this is right, won't use it below

cnn_triu_dense = layer_to_flattened_triu(cnn_dense)
# conv1_triu = (cnn.iloc[:, :1792])
lstm_triu_dense = layer_to_flattened_triu(lstm_dense)

column_names = ['w2v', 'Tfidf', 'CNN', 'LSTM', 'Human']
df_norm = pd.DataFrame([normalize(w2v_triu),normalize(tfidf_triu), normalize(cnn_triu_dense), normalize(lstm_triu_dense), normalize(human_triu)])
df_norm_corr = df_norm.T.corr(method='spearman')
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'similarity_experiment_heatmap_normalized_dense_spearman_sorted', with_corr_values=True)

df_norm_corr = df_norm.T.corr(method='pearson')
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'similarity_experiment_heatmap_normalized_dense_pearson_sorted', with_corr_values=True)







# for 6-way classification model
# ========================



categories = ['University','Decoration','MilitaryConflict','MilitaryPerson','Politician', 'Monarch']
Xtest, Ytest = data_helpers.load_data(config.test_path, categories)

Xtest_all, Ytest_all= data_helpers.load_data(config.test_path, config.categories)

# Xtest_raw, Ytest_raw = data_helpers.load_data_raw(config.test_path, categories)

# convert stimuli id for 64*1200 to 6*1200
stimuli_6 = [73313-73200+categories.index('University')*1200,
 73378-73200+categories.index('University')*1200,
 49222-49200+categories.index('Politician')*1200,
 50278-49200+categories.index('Politician')*1200,
 38088-37200+categories.index('MilitaryPerson')*1200,
 37198-36000+categories.index('MilitaryConflict')*1200,
 38031-37200+categories.index('MilitaryPerson')*1200,
 73423-73200+categories.index('University')*1200,
 41201-40800+categories.index('Monarch')*1200,
 22386-21600+categories.index('Decoration')*1200,
 22709-21600+categories.index('Decoration')*1200,
 21878-21600+categories.index('Decoration')*1200,
 36885-36000+categories.index('MilitaryConflict')*1200,
 38039-37200+categories.index('MilitaryPerson')*1200,
 41080-40800+categories.index('Monarch')*1200,
 37053-36000+categories.index('MilitaryConflict')*1200,
 49534-49200+categories.index('Politician')*1200,
41311-40800+categories.index('Monarch')*1200]


stimuli_6 = np.sort(stimuli_6)

# LogReg w2v

input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_w2v_6_final/'
w2v_6 = pd.DataFrame(np.load(input_dir+'probability_output.npy')[stimuli_6], index=stimuli_6)


# LogReg tfidf
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/logreg_tfidf_6_final/'
tfidf_6 = pd.DataFrame(np.load(input_dir+'probability_output.npy')[stimuli_6], index=stimuli_6)
# tfidf = pd.DataFrame(np.loadtxt(input_dir+'probability_output.txt', delimiter=' ', dtype = str)[stimuli], index=stimuli)

# CNN softmax
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_6/'
cnn_dense_6 = pd.DataFrame(np.load(input_dir+'output_layers.npz')['f'][stimuli_6], index=stimuli_6)

# LSTM CNN softmax
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/lstm18_6/'
# lstm= pd.DataFrame(np.load(input_dir+'output_layers_softmax_final.npz')['a'][stimuli], index=stimuli)
lstm_dense_6 = pd.DataFrame(np.load(input_dir+'output_layers_dense_final.npz')['a'][stimuli_6], index=stimuli_6)


# RSM of each model on 6.
# ======================================================
method = 'spearman'
column_names =columns
#  column_names = list(range(1,19))
plt.clf()
plt.figure(1, figsize=(10,10))
plt.subplot(221)
w2v_corr = pd.DataFrame(normalize(w2v_6.T.corr(method=method).values))
subplot_heatmap(w2v_corr, False, True, column_names)
plt.title('Averaged word embeddings')
plt.ylabel([])
plt.subplot(222)
tfidf_corr = pd.DataFrame(normalize(tfidf_6.T.corr(method=method).values))
subplot_heatmap(tfidf_corr, False, False, column_names)
plt.title('Tf-idf')
plt.subplot(223)
cnn_corr = pd.DataFrame(normalize(cnn_dense_6.T.corr(method=method).values))
subplot_heatmap(cnn_corr, True, True, column_names)
plt.title('CNN')
plt.subplot(224)
lstm_corr = pd.DataFrame(normalize(lstm_dense_6.T.corr(method=method).values))
subplot_heatmap(lstm_corr , True, False, column_names)
plt.title('LSTM')
# plt.subplot(313, sharex=ax1, sharey=ax1)
# plt.tight_layout()
plt.savefig(output_dir+'rsm_of_models_6_sorted'+ '.eps', format='eps', dpi=100)




# Correlate RSMs
# ==================================================================================================================================
w2v_triu_6 = layer_to_flattened_triu(w2v_6)
tfidf_triu_6 = layer_to_flattened_triu(tfidf_6)
cnn_triu_6 = layer_to_flattened_triu(cnn_dense_6)
# conv1_triu = (cnn.iloc[:, :1792])
lstm_triu_6 = layer_to_flattened_triu(lstm_dense_6)

column_names = ['w2v', 'Tfidf', 'CNN', 'LSTM', 'Human']
df_norm = pd.DataFrame([normalize(w2v_triu),normalize(tfidf_triu_6), normalize(cnn_triu_6), normalize(lstm_triu_6), normalize(human_triu)])
df_norm_corr = df_norm.T.corr(method=method)
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'similarity_experiment_heatmap_normalized_dense_spearman_6_sorted', with_corr_values=True)

df_norm_corr = df_norm.T.corr(method='pearson')
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'similarity_experiment_heatmap_normalized_dense_pearson_6_sorted', with_corr_values=True)




# count
from collections import Counter

input_dir= '/Users/danielmlow/Dropbox/cnn/mturk/semantic_similarity/results/'
df = pd.read_csv(input_dir+'full.csv')
total = len(set(list(df._worker_id)))
judg_per_worker = list(Counter(list(df._worker_id)).values())

input_dir= '/Users/danielmlow/Dropbox/cnn/mturk/categorization/results/'
df = pd.read_csv(input_dir+'full.csv')
total = len(set(list(df._worker_id)))
judg_per_worker = list(Counter(list(df._worker_id)).values())
np.mean(judg_per_worker )
np.std(judg_per_worker )




#
# plt.clf()
# plt.hist(human_triu, bins=10)
# plt.savefig(output_dir+'temp2')
#
#
#
#
# # lstm_triu2 = standardize(np.array(lstm_triu).reshape(-1,1))
#
#
#
#
#
# # Compare RSM trius
# # ================================================================================================
# human_triu = normalize(human_triu)
# lstm_triu = normalize(lstm_triu)
# cnn_triu = normalize(cnn_triu)
#
# print(pearsonr(human_triu, lstm_triu)[0].round(2),pearsonr(human_triu, cnn_triu)[0].round(2))
#
#
# # Plot RSMs for each model and for human
#
#
#
#
# # Obtain logreg coefficients for tfidf model. Each sentence is
# # Obtain tf-idf representation for each sentence
# # ======================================================================================================
# tf = TfidfVectorizer()
# scores = []
# tfidf_matrix = tf.fit_transform(Xtest)
# feature_names = tf.get_feature_names()
#
# feature_vectors = []
# # Replace iterator for ids you want, which are rows in the matrix.
# for sentence in range(len(np.array(Xtest)[stimuli])):
#     feature_ids = tfidf_matrix[stimuli[sentence], :].nonzero()[1]
#     feature_vector = [tfidf_matrix[sentence,n] for n in feature_ids]
#     feature_vectors.append(feature_vector)
#
# # They all have different sizes, so choose the highest N, where N is the min amount that all features  have.
# lengths = []
# for i in feature_vectors:
#     lengths.append(len(i))
#
# min_amount_of_features = np.min(lengths)
#
# feature_vectors0 = []
# for i in feature_vectors:
#     print(i)
#     vector_sorted = np.sort(i)
#
#
# #Most sentences just have 0-3 words with high tfidf. Reduce to 3 dimensions to do RSM:
# from sklearn.decomposition import PCA
# from numpy import linalg as LA
# import config as cfg
#
# cov = np.cov(tfidf_matrix)#cov of features.
# w, v = LA.eig(cov) #eigenvalue decomposition
#
# X = np.array(cov)
# pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
#


# Example of sentences