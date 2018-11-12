'''
Daniel M. Low

Edited Nov. 11 2018
'''


import pandas as pd
import numpy as np
import config
import data_helpers
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

from numpy.random import seed
seed(123)


def Nsentences(n, layer, amount_sent_per_category):
    layer_sample = pd.DataFrame()
    for i in range(0,layer.shape[0], amount_sent_per_category):
        n_sentences_1category = layer.iloc[i:i+amount_sent_per_category]
        n_sentences_1category = n_sentences_1category.sample(frac=1).iloc[:n]
        layer_sample = layer_sample.append(n_sentences_1category)
    return layer_sample


def count_sent_len_per_category(amount_sent_per_category, Xvalidation, categories):
    df = pd.DataFrame(index=range(0,45))
    lengths_all_categories = pd.DataFrame(index=range(amount_sent_per_category))
    for i in range(0,len(Xvalidation), amount_sent_per_category):
        # loop through sentences in categories
        lengths = []
        for j in range(i,i+amount_sent_per_category):
            length = len(Xvalidation[j].split())
            lengths.append(length)
        lengths_all_categories = pd.concat([lengths_all_categories, pd.DataFrame(lengths, index=range(amount_sent_per_category))],axis=1)
    for z in range(len(categories)):
        df = pd.concat([df, lengths_all_categories.iloc[:,z].value_counts()], axis=1)
    df.columns = categories
    return df



# Main
# ====================================================================================================================
punc = re.compile('[%s]' % re.escape(string.punctuation))
categories = config.categories
statistic = 'spearman'
# model_name = 'cnn35_64'
# path_to_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/'+model_name+'/'
# input_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/stimuli/'
# output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/stimuli_final/'




# Main
# =======================================================================================================================

# Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=True) # I changed this so it combines train and validation




# Make or load. Make:
# df_prototypical, df_prototypical_score, df_prototypical_sentences = corr_between_layers.prototypical_sentences(statistic, X,X_raw, path_to_dir, layer=layers[-2], validation_size= len(X), amount_sent=int(len(X)/len(categories)),nan='', categories=categories)

# Load:
# df_prototypical_sentences = pd.read_csv(path_to_dir+'df_prototypical_sentences_raw_spearman.csv')[categories] #TODO make sure you want sentences_raw
# df_prototypical = pd.read_csv(path_to_dir+'df_prototypical_spearman.csv')
# df_prototypical.columns = categories
# df_prototypical_scores = pd.read_csv(path_to_dir+'df_prototypical_score_spearman.csv')
# df_prototypical_scores.columns = categories

# df_prototypical = pd.read_csv(path_to_dir+'df_prototypical_'+statistic+'.csv') #TODO: reorder column list
# df_prototypical.columns = categories
# df_prototypical_sentences= pd.read_csv(path_to_dir+'/df_prototypical_sentences_'+statistic+'.csv')[categories]


# What is CD count
# log freq (base 10)?
# did it import accents correctly?
# We can either lowecase first word or lowercase all (but then proper nouns like Peter will have freq=0).
# Lowe case only first character category1_lower = [n[0].lower() for n in list(category1)]


freq_dir = '/Users/danielmlow/Dropbox/cnn/thesis/'
freq0 = pd.read_csv(freq_dir+'subtlex-it.csv').iloc[:,1:3]


counts = list(freq0.iloc[:,1])
counts_plus_1= [x+1 for x in counts]
log_counts = [np.log10(n) for n in counts_plus_1]
freq0['logFREQcount'] = log_counts

logfreq = pd.DataFrame(list(freq0.iloc[:,2]), index=list(freq0.iloc[:,0])).T



with open('stopwords-it2.txt', 'r') as f:
    sw = f.readlines()

italian_stop_words = [n.replace('\n','') for n in sw]
italian_stop_words.append('d')




def filter_sentences_between_lengths(min_length=14, max_length=15, df_sentences=None, df_ids=None, df_prototypical_scores=None, amount_sent_per_category=None, top_n_sentences=False):
    if top_n_sentences!=False:
        df_sentences = df_sentences.iloc[:top_n_sentences,:]
    filtered_sentences = pd.DataFrame(np.zeros([df_sentences.shape[0],1]))
    filtered_ids = pd.DataFrame(np.zeros([df_sentences.shape[0], 1]))
    # filtered_scores = pd.DataFrame(np.zeros([df_sentences.shape[0], 1]))
    mean_frequencies = [] #for histogram without zeros
    content_words = [] #for histogram
    for column in range(len(categories)):
        valid_sentence_1category = []
        valid_id_1category = []
        # valid_score_1category = []
        for row in range(0,df_sentences.shape[0]):
            sentence = df_sentences.iloc[row, column]
            sentence_no_punctuation = punc.sub('', sentence)  # removes punctuation, not accents
            sentence_no_stop_words = [n for n in sentence_no_punctuation.split() if n not in italian_stop_words]
            sentence_frequencies = []
            for word in sentence_no_stop_words:
                try:
                    frequency = int(logfreq[word]) #log10(frequency+1) so freq=1 isn't 0, but ~0.3
                except:
                    frequency = 0 # if freq == 0, then remove sentence
                    pass
                sentence_frequencies.append(frequency)
            length = len(sentence.split())
            # min_freq = np.argmin(sentence_frequencies)
            mean_freq = np.mean(sentence_frequencies)
            mean_frequencies.append(mean_freq)
            content_words.append(len(sentence_frequencies))
            # if (length >= min_length and length <= max_length and not (any(char.isdigit() for char in sentence)) and (mean_freq >= 0.86 and mean_freq <= 3.09)):
            if (length >= min_length and length <= max_length and (mean_freq >= 1.125 and mean_freq <= 2.875)): #run it once, see Q1 and Q3 and define np.percentile(mean_frequencies, 5)
                id = df_ids.iloc[row, column]
                # score = df_prototypical_scores.iloc[row, column]
                valid_sentence_1category.append(sentence)
                valid_id_1category.append(int(id))
                # valid_score_1category.append(np.round(score,2))
        filtered_sentences[column]=pd.DataFrame(valid_sentence_1category, index=None)
        filtered_ids[column] = pd.DataFrame(valid_id_1category, index=None)
        # filtered_scores[column] = pd.DataFrame(valid_score_1category, index=None)
    filtered_sentences.columns = categories
    filtered_ids.columns = categories
    # filtered_scores.columns = categories
    return filtered_sentences, filtered_ids, mean_frequencies, content_words



def test_length(path = '/Users/danielmlow/Dropbox/cnn/experiment/final_experiment_daniel/sentences_category_id/d2c_cleaned.csv'):
    sentences = pd.read_csv(path)['sentences']
    for sentence in sentences:
        length = len(sentence.replace("'", " ").split()) #replace apostrophes
        if not length >= 14 and length <= 15:
            print(length)
            print(sentence)

test_length('/Users/danielmlow/Dropbox/cnn/experiment/final_experiment_daniel/sentences_category_id/d2c_cleaned.csv')

def insert_columns(prototypical_sentences_10_14):
    prototypical_sentences_8_14_translations = prototypical_sentences_10_14[:]
    for i in range(1, len(categories) * 2 + 1, 2):
        prototypical_sentences_8_14_translations.insert(i, column='translation_' + categories[int(i / 2)],
                                                        value=np.zeros((300, 1)))
    return prototypical_sentences_8_14_translations




def clean(s):
    to_remove = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    for i in to_remove:
        s = s.replace(i, "")
    s = re.sub(r"\s{2,}", " ", s)
    return s

def lower_first_character(sentences):
    processed = [str(n)[0].lower()+str(n)[1:] for n in sentences]
    return processed




def clean_apostr(s):
    s = s.replace('L a', "L'a")
    s = s.replace('L e', "L'e")
    s = s.replace('L i', "L'i")
    s = s.replace('L o', "L'o")
    s = s.replace('L u', "L'u")
    s = s.replace('l a', "l'a")
    s = s.replace('l e', "l'e")
    s = s.replace('l i', "l'i")
    s = s.replace('l o', "l'o")
    s = s.replace('l u', "l'u")
    s = s.replace('l A', "l'A")
    s = s.replace('l E', "l'E")
    s = s.replace('l I', "l'I")
    s = s.replace('l O', "l'O")
    s = s.replace('l U', "l'U")
    s = s.replace('dov e', "dov'e")
    s = s.replace('com e', "com'e")
    s = s.replace('c e', "c'e")
    s = s.replace('C e', "C'e")
    s = s.replace(':', "")
    s = re.sub(r"\s{2,}", " ", s)
    return s




def lower_first_character_df(df_sentences, categories):
    cleaned_filtered_sentences = []
    for category in range(len(categories)):
        category1 = df_sentences.iloc[:,category]
        category1_lower = [str(n)[0].lower()+str(n)[1:] for n in list(category1)]
        cleaned_filtered_sentences.append(category1_lower)
    cleaned_filtered_sentences = pd.DataFrame(cleaned_filtered_sentences).T
    cleaned_filtered_sentences.columns = categories
    return cleaned_filtered_sentences

def Nsentences_Klength(n, sentence_length, layer, amount_sent_per_category, X):
    layer_sample = pd.DataFrame()
    for i in range(0,layer.shape[0], amount_sent_per_category):
        n_sentences_same_length = pd.DataFrame()
        sentences = 0
        # loop through sentences in categories
        for j in range(i,i+amount_sent_per_category):
            if len(X[j].split())==sentence_length:
                n_sentences_same_length = n_sentences_same_length.append(layer.T[j])
                sentences+=1
                if sentences == n:
                    break
        if n_sentences_same_length.shape[0] < n:
            print('not enough sentences of that length. Try again.')
            break
        else:
            layer_sample = layer_sample.append(n_sentences_same_length)
    return layer_sample




# def replace_numbers(s):
#     DIGITS = re.compile("[0-9]", re.UNICODE)
#     s = DIGITS.sub("#", s)
#     return s


# Create csv with output layers for test set
# ============================================================================================================
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all/'

categories = config.categories

whole_dataset=True

if whole_dataset:
    X, y= data_helpers.load_whole_dataset(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)
    X_raw, y= data_helpers.load_whole_dataset_raw(config.train_path, config.validation_path, config.test_path,categories,load_all=True, shuffle=False,one_hot=False)
    conv_1 = pd.DataFrame(np.load(input_dir+'output_layer_conv_1.npz')['a'])
    # conv_2 = pd.DataFrame(np.load('/Volumes/Maxtor/final_model/output_layers/output_layer_c.npz')['a'])
    dense_final = pd.DataFrame(np.load(input_dir + 'output_layer_dense_final.npz')['a'])
    output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/whole_dataset/'
    output_dir2 = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/whole_dataset/other/'
    sentences_per_category = 6000
else:
    ## Just test set
    # X, Ytest = data_helpers.load_data(config.test_path, categories)
    X_raw, Ytest_raw = data_helpers.load_data_raw(config.test_path, categories)
    layers = data_helpers.load_output_layers(input_dir)
    conv_1 = layers[0]
    dense_final= layers[-2]
    output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/just_test_set/'
    output_dir2 = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/just_test_set/other/'
    sentences_per_category = 1200

# Create df with all sentences per category (sentences_per_categoryx64)
# Create on raw dataset. Count contractions as 2 words because that's what the model does, because they each have their own word embedding and freq dict seperates them.


ids = list(range(sentences_per_category*64))
ids = pd.DataFrame(np.array(ids).reshape(64,sentences_per_category)).T
ids.columns = categories


X_raw_cleaned = [clean(n) for n in X_raw] #this is so it can calculate freq
X_raw_cleaned2 = lower_first_character(X_raw_cleaned) #calculate freq on first word lower_cased

X_raw_df  = pd.DataFrame(np.array(X_raw_cleaned2).reshape(64,sentences_per_category)).T
X_raw_df.columns = categories

# Lets shuffle so when we filter A)we obtain a random selection from train and test set and B) there are sentences with max eg 384 nonNAN values, so we want to randomly choose sentences that have more.
ids_shuffled = ids.sample(frac=1)
X_raw_df_shuffled = pd.DataFrame()
for column in range(64):
    category_column = list(X_raw_df.iloc[:, column])
    new_order = list(ids_shuffled.iloc[:, column])
    new_order = [x - (sentences_per_category*column) for x in new_order]
    column_shuffled = [category_column[i] for i in new_order]
    X_raw_df_shuffled[categories[column]] = column_shuffled

# Make sure it's correct
print(X_raw_df_shuffled.iloc[1000,30] == X_raw_cleaned2[ids_shuffled.iloc[1000,30]])
print(X_raw_df_shuffled.iloc[20,0] == X_raw_cleaned2[ids_shuffled.iloc[20,0]])

# Filter sentences
# ====================================================================================================================
filtered_sentences, filtered_ids, mean_frequencies, content_words = filter_sentences_between_lengths(min_length=14, max_length=15, df_sentences=X_raw_df_shuffled, df_ids=ids_shuffled, df_prototypical_scores=None, amount_sent_per_category=sentences_per_category, top_n_sentences=False)

print(np.nanpercentile(mean_frequencies, 5).round(4))
print(np.nanpercentile(mean_frequencies, 95).round(4))

# filtered_sentences, filtered_ids, mean_frequencies, content_words = filter_sentences_between_lengths(min_length=14, max_length=15, df_sentences=X_raw_df_shuffled, df_ids=ids_shuffled, df_prototypical_scores=None, amount_sent_per_category=sentences_per_category, top_n_sentences=False)

filtered_sentences.to_csv(output_dir2+'filtered_sentences'+'_14_15_ALL.csv', index=False, encoding='utf-8-sig', header=True)
filtered_ids.to_csv(output_dir2+'filtered_ids'+'_14_15_ALL.csv', index=False, header=True)

filtered_sentences1 = filtered_sentences.dropna()
filtered_ids1 = filtered_ids.dropna()
filtered_ids2 = filtered_ids1.astype(int)

filtered_sentences1.to_csv(output_dir+'filtered_sentences'+'_14_15.csv', index=False, encoding='utf-8-sig', header=True)
filtered_ids2.to_csv(output_dir+'filtered_ids'+'_14_15.csv', index=False, header=True)


# Compute statistics and plots on freq and content words
# ==========================================================================================
plt.clf()
plt.hist(mean_frequencies, bins=150)
plt.xlabel('log(lexical frequency + 1)')
plt.ylabel('Amount of sentences')
plt.savefig(output_dir2+'mean_frequencies.png')

plt.clf()
plt.hist(content_words, bins=32)
plt.xlabel('Amount of content words')
plt.ylabel('Amount of sentences')
plt.savefig(output_dir2+'content_words.png')

from scipy.stats import describe

content = describe(content_words)
frequencies = describe(mean_frequencies)

with open(output_dir2+'descriptive_statistics.txt', 'w') as f:
    f.write('statistics on content words per sentence:'+'\n')
    f.write(str(content)+'\n')
    f.write('percentile 5, remove lower:'+'\n')
    f.write(str(np.percentile(mean_frequencies, 5).round(4))+'\n')
    f.write('percentile 95, remove higher:' + '\n')
    f.write(str(np.percentile(mean_frequencies, 95).round(4)) + '\n')
    f.write('statistics on log(mean(frequency +1)) per sentence:'+'\n')
    f.write(str(frequencies)+'\n')

# Clustermap of single sentences: Make sure vectors are in the right order
# ========================================================================================================
output_dir2 = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/whole_dataset/other/'

layer_name = 'conv_1'

random_ids = np.random.randint(0,76800, 5)
random_sentences = np.array(X_raw_cleaned2)[random_ids]
words_per_sentence = []
for i in random_sentences:
    length = len(i.split())
    words_per_sentence.append(length)

conv1_layers = []
for i in random_ids:
    vector =conv_1.iloc[i,:]
    conv1_layers.append(vector)

conv1_layers = np.array(conv1_layers)

for i in range(len(conv1_layers)):
    single_sentence_conv1 = pd.DataFrame(np.reshape(conv1_layers[i], (36, 128)))
    plt.clf()
    cg = sns.heatmap(single_sentence_conv1, cmap="RdBu_r", vmin=-1., vmax=1.,
                                cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
    fig = cg.get_figure()
    plt.title(random_sentences[i]+' \nwords with embeddings: '+str(words_per_sentence[i]),fontsize=5)
    fig.savefig(output_dir2+ 'single_sentence_'+str(random_ids[i])+'.eps', format='eps', dpi=100)



# Build RDM
# ===================================================================================================================
layer_names = ['conv_1', 'pool_1', 'conv_2', 'pool_2', 'dense_1', 'dense_final', 'softmax_final']

if whole_dataset:
    layer = conv_1[:]
else:

    layer = layers[0]

layer_name = layer_names[0]

print(layer_name)

amount_sent_per_category = int(len(X_raw_cleaned2)/len(categories))
n = 5
k = [14, 15]
layer_sample_small = Nsentences_Klength(n, k[0], layer, amount_sent_per_category, X_raw_cleaned2)
layer_sample_large = Nsentences_Klength(n, k[1], layer, amount_sent_per_category, X_raw_cleaned2)
statistic = 'pearson'
sentences_from_each = len(categories)*n
layer_sample = pd.concat([layer_sample_small, layer_sample_large])
df = pd.DataFrame(layer_sample)
df = df.T.corr(method=statistic) #TODO: should I transpose like this?
df.index= range(0,sentences_from_each*2)
df.columns= range(0,sentences_from_each*2)
df_triu = pd.DataFrame(np.triu(df, k=1)).replace(0, np.nan)

zeros = pd.DataFrame(np.ones([sentences_from_each,sentences_from_each]))
a15 = pd.DataFrame(np.full([sentences_from_each, sentences_from_each], 20))
sentence_len1 = pd.concat([zeros, a15])
sentence_len2 = pd.concat([a15, zeros])
sentence_len = pd.concat([sentence_len1 ,sentence_len2], axis=1)
sentence_len.index=range(sentences_from_each*2)
sentence_len.columns=range(sentences_from_each*2)
sentence_len_triu= pd.DataFrame(np.triu(sentence_len, k=0)).replace(0, np.nan)
sentence_len_triu = sentence_len_triu.replace(1,0)
print('Sentence_len_triu_dist effect ' + layer_name + ' ' + str(k) + ': \n')
print(str(sentence_len_triu.corrwith(df_triu).mean()) + '\n')


# Save layer
# ===================================================================================================================

output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/whole_dataset/'

filtered_ids2 = pd.read_csv(output_dir+'filtered_ids_14_15.csv')
filtered_sentences1 = pd.read_csv(output_dir+'filtered_sentences_14_15.csv')

filtered_sentences_flatten = filtered_sentences1.values.T.flatten()
filtered_ids_flatten = filtered_ids2.values.T.flatten()

import scipy.io as sio

sio.savemat(output_dir+'filtered_sentences_list.mat', {'sentences':filtered_sentences_flatten})
sio.savemat(output_dir+'filtered_ids_list.mat', {'ids':filtered_ids_flatten})

conv_1= pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all_2/output_layers_whole_dataset_a.npz')['a'])
pool_1 = pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all_2/output_layers_whole_dataset_b.npz')['a'])
conv_2= pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all_2/output_layers_whole_dataset_c.npz')['a'])
pool_2 = pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all_2/output_layers_whole_dataset_d.npz')['a'])
dense_final = pd.DataFrame(np.load('/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all_2/output_layers_whole_dataset_f.npz')['a'])

conv_1_filtered = conv_1.loc[filtered_ids_flatten]
pool_1_filtered = pool_1.loc[filtered_ids_flatten]
conv_2_filtered = conv_2.loc[filtered_ids_flatten]
pool_2_filtered = pool_2.loc[filtered_ids_flatten]
dense_final_filtered = dense_final.loc[filtered_ids_flatten]

conv_1_filtered.to_csv(output_dir+'conv_1_filtered_vectors.csv')
pool_1_filtered.to_csv(output_dir+'pool_1_filtered_vectors.csv')
conv_2_filtered.to_csv(output_dir+'conv_2_filtered_vectors.csv')
pool_2_filtered.to_csv(output_dir+'pool_2_filtered_vectors.csv')
dense_final_filtered.to_csv(output_dir+'dense_final_filtered_vectors.csv')



a = pd.read_csv(output_dir+'pool_2_filtered_vectors.csv')


subsample = conv_1_filtered.sample(10)

sentences_id = list(subsample.index)
sentences = [X_raw_cleaned2[i] for i in sentences_id]
sentences_len = [len(n.split()) for n in sentences]

plt.clf()
sns.set(font_scale = 0.3)
cg = sns.heatmap(subsample, cmap="RdBu_r", vmin=-1., vmax=1.,
                 cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
fig = cg.get_figure()
fig.savefig(output_dir2 + 'heatmap_of_single_sentences' + str(random_ids[i]) + '.eps', format='eps', dpi=100)



# Santiy check
# ====



def Nsentences(n, layer, amount_sent_per_category):
    layer_sample = pd.DataFrame()
    for i in range(0, layer.shape[0], amount_sent_per_category):
        n_sentences_1category = layer.iloc[i:i + amount_sent_per_category]
        n_sentences_1category = n_sentences_1category.sample(frac=1).iloc[:n]
        layer_sample = layer_sample.append(n_sentences_1category)
    return layer_sample


# RSA single sentences
# =====================================================================================================================================
amount_sent_per_category = int(len(Xtest) / len(categories))
n = 5

amount_sent_per_category=332

layer_names = ['conv_1', 'pool_1', 'conv_2', 'pool_2', 'dense_1', 'dense_final', 'softmax_final']
layer = dense_final_df[:]



layer_name = layer_names[-2]

for layer, layer_name in zip(layers, layer_names):
    # layer_sample = Nsentences_a_k_length(n, 10, 14, layer, amount_sent_per_category, Xtest)
    layer_sample = Nsentences(n, layer, amount_sent_per_category)

    # Clustermap
    df = pd.DataFrame(layer_sample)
    # df[(df >= -0.08) & (df <= 0.09)] = np.nan
    statistic = 'spearman'
    df = df.T.corr(method=statistic)
    print('done')
    columns = [[i] * n for i in categories]  # TODO: categories or categories_wrong.
    # columns = [[i]*n for i in config.categories_wrong]
    columns = [i for j in columns for i in j]
    df.columns = columns
    df.index = columns
    sns.set(font_scale=0.08)
    cg = sns.clustermap(df, method='ward', cmap="RdBu_r", vmin=-1., vmax=1.,
                        cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, )
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # tick_locator = ticker.MaxNLocator(int(df.shape[0]))
    # plt.setp(cg.ax_heatmap.xaxis.set_major_locator(tick_locator))
    # plt.setp(cg.ax_heatmap.yaxis.set_major_locator(tick_locator))

    cg.savefig(output_dir + 'RSA_ward_clustermap' + layer_name + '_single_sentences_' + '.eps', format='eps', dpi=100)

    cg.savefig(path_to_dir + '/RSA_ward_clustermap_' + layer_name + '_single_sentences_' + statistic + '.eps',
               format='eps', dpi=100)





# 1. CLustering
# ================================================================================================
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score



def cluster(n_clusters, layer, output_dir, file_name):
    clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(layer)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # silhouette_avg = silhouette_score(layer, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(layer, cluster_labels)
    # silh_values.append(sample_silhouette_values)

    category = [[n]*332 for n in range(64)]
    category = [n for i in category for n in i]
    cluster_df = pd.DataFrame({'ids':ids, 'category':category, 'cluster_labels':cluster_labels, 'silh_values':sample_silhouette_values})
    cluster_df.to_csv(output_dir+'cluster_df_'+file_name+'.csv')
    return cluster_labels, sample_silhouette_values

output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/'
dense_final_df = pd.read_csv(output_dir+'whole_dataset/dense_final_filtered_vectors.csv', index_col=0)
dense_final = np.array(dense_final_df)
conv_1_df = pd.read_csv(output_dir+'whole_dataset/conv_1_filtered_vectors.csv', index_col=0)
conv_1 = np.array(conv_1_df)
ids = np.array(dense_final_df.index)



# Scotts method to clustering
# ==================================================================================================================
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




# TODO: measure of how well it matches original labels.



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

layer = dense_final[:]
n_clusters = 64

file_name = 'd2c'
cluster_labels, sample_silhouette_values = cluster(n_clusters, layer, output_dir, file_name)
# #
np.save(output_dir+'cluster_labels_'+file_name,cluster_labels)
np.save(output_dir+'sample_silhouette_values_'+file_name, sample_silhouette_values)

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


# sanity check
# sent = final_sentences[1][1]
# value= final_sentences_values[1][1]
# print(value == corr_layerB.loc[sent][:20].mean()-corr_layerB.loc[sent][20:].mean())
# sent = final_sentences[44][5]
# value= final_sentences_values[44][5]
# print(value == corr_layerB.loc[sent][:20].mean()-corr_layerB.loc[sent][20:].mean())



final_sentences_d2c = [n for i in final_sentences for n in i]

corr_layer1_all = layer1.iloc[:,:1792].loc[final_sentences_d2c].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences_d2c].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))
# 0.4207
# max= 0.614
np.save(output_dir+'final_ids_d2c.npy', final_sentences_d2c)


# c2d
# ========================================================================================================================
layer = conv_1_df.iloc[:,:1792]
n_clusters = 64

file_name = 'c2d'
cluster_labels, sample_silhouette_values = cluster(n_clusters, layer, output_dir, file_name)
#
np.save(output_dir+'cluster_labels_'+file_name,cluster_labels)
np.save(output_dir+'sample_silhouette_values_'+file_name, sample_silhouette_values)

cluster_labels = np.load(output_dir+'cluster_labels_c2d.npy')
sample_silhouette_values = np.load(output_dir+'sample_silhouette_values_c2d.npy')
# create dictionary {cluster_0: [silh_value_id0, ...],...


d_silh, d_ids = dict_silh(cluster_labels, sample_silhouette_values)
d_highest_silh, d_highest_ids = rank_silh(d_silh, d_ids)

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

corr_layer1_all = layer1.loc[final_sentences_c2d].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences_c2d].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))
# 0.5683
np.save(output_dir+'final_ids_c2d.npy', final_sentences_c2d)


# Combine both
# ====================================================================
final_ids_c2d = np.load(output_dir+'final_ids_c2d.npy')
final_ids_d2c = np.load(output_dir+'final_ids_d2c.npy')

corr_conv_1_c2d = conv_1_df.loc[final_ids_c2d ].iloc[:, :1792].T.corr()
corr_conv_1_d2c = conv_1_df.loc[final_ids_d2c].iloc[:, :1792].T.corr()
corr_df_c2d = dense_final_df.loc[final_ids_c2d ].T.corr()
corr_df_d2c = dense_final_df.loc[final_ids_d2c].T.corr()

corr_conv_1_c2d_triu =  corr_conv_1_c2d.where(np.triu(np.ones(corr_conv_1_c2d.shape)).astype(np.bool)).replace(1,np.nan).values.flatten()
corr_conv_1_c2d_triu  = corr_conv_1_c2d_triu [~np.isnan(corr_conv_1_c2d_triu )][:73471]

corr_conv_1_d2c_triu =corr_conv_1_d2c.where(np.triu(np.ones(corr_conv_1_d2c.shape)).astype(np.bool)).replace(1,np.nan).values.flatten()
corr_conv_1_d2c_triu = corr_conv_1_d2c_triu[~np.isnan(corr_conv_1_d2c_triu )]

corr_df_c2d_triu = corr_df_c2d.where(np.triu(np.ones(corr_df_c2d.shape)).astype(np.bool)).replace(1,np.nan).values.flatten()
corr_df_c2d_triu = corr_df_c2d_triu [~np.isnan(corr_df_c2d_triu)][:73471]

corr_df_d2c_triu = corr_df_d2c.where(np.triu(np.ones(corr_df_d2c.shape)).astype(np.bool)).replace(1,np.nan).values.flatten()
corr_df_d2c_triu = corr_df_d2c_triu [~np.isnan(corr_df_d2c_triu )]


plt.scatter(corr_df_d2c_triu,corr_conv_1_d2c_triu,c='blue', label='d2c', s=20 )
plt.scatter(corr_df_c2d_triu,corr_conv_1_c2d_triu,c='green', label='c2d', s=20, marker='o')

plt.xlabel('df')
plt.ylabel('conv1')
plt.legend()
plt.show()


corr_layer1_all = layer1.loc[final_sentences_c2d].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences_c2d].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)



import matplotlib.pyplot as plt
plt.scatter(df_d2c,conv_1_d2c )
plt.scatter()



conv_1_all = conv_1_df.loc[np.concatenate((final_ids_c2d,final_ids_d2c))].iloc[:, :1792]
dense_final_all = dense_final_df.loc[np.concatenate((final_ids_c2d,final_ids_d2c))]

corr_layer1_all = conv_1_all.T.corr(method=method)
corr_layer3_all = dense_final_all.T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan).reset_index(drop=True)
corr_layer1_all_triu.columns = range(corr_layer1_all_triu.shape[1])
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan).reset_index(drop=True)
corr_layer3_all_triu.columns = range(corr_layer3_all_triu.shape[1])
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))
# 0.521

print('1. correlation between triu of conv1 from c2d and DF from c2d: ', 0.568)
print('2. correlation between triu of conv1 from d2c and DF from d2c: ', 0.421)
print('3. correlation between triu of conv1 from c2d and DF from d2c: ', 0.521)



# Import Scotts sentences and try
# ==========================================================================================
import scipy.io as sio
output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/'
scott_ids = sio.loadmat(output_dir+'vecForDL.mat')

manX = scott_ids.get('manX')[0]
manY = scott_ids.get('manY')[0]

from scipy.stats import pearsonr
pearsonr(manX, manY)

import scipy.io as sio
# scott_ids = sio.loadmat(output_dir+'indsForDaniel.mat')
scott_ids = sio.loadmat(output_dir+'forDaniel2.mat')
scott_ids = sio.loadmat(output_dir+'forDLv3.mat')


scott_ids.keys()

d2c = scott_ids.get('d2c')[0]

final_sentences =[n-1 for n in d2c]

corr_layer1_all_d2c = conv_1_df.iloc[final_sentences,:1792].T.corr(method=method)
corr_layer3_all_d2c = dense_final_df.iloc[final_sentences].T.corr(method=method)

corr_layer1_all_triu_d2c = corr_layer1_all_d2c.where(np.triu(np.ones(corr_layer1_all_d2c.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu_d2c = corr_layer3_all_d2c.where(np.triu(np.ones(corr_layer3_all_d2c.shape)).astype(np.bool)).replace(1,np.nan)
means3 = corr_layer3_all_triu_d2c.mean().mean()
means1 = corr_layer1_all_triu_d2c.mean().mean()

corr_between_layers_d2c = corr_layer1_all_triu_d2c.corrwith(corr_layer3_all_triu_d2c).mean()

print(corr_between_layers_d2c.round(6))


0.498962


# 0.342474


# Load clustered from l1 to l3

c2d = scott_ids.get('c2d')[0]
final_sentences =[n-1 for n in c2d]

corr_layer1_all_c2d = conv_1_df.iloc[final_sentences,:1792].T.corr(method=method)
corr_layer3_all_c2d = dense_final_df.iloc[final_sentences].T.corr(method=method)
corr_layer1_all_triu_c2d = corr_layer1_all_c2d.where(np.triu(np.ones(corr_layer1_all_c2d.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu_c2d = corr_layer3_all_c2d.where(np.triu(np.ones(corr_layer3_all_c2d.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers_c2d = corr_layer1_all_triu_c2d.corrwith(corr_layer3_all_triu_c2d).mean()

print(corr_between_layers_c2d.round(3))
# 0.461235

# Correlate both sentences

# -0.10257081395724943

from scipy.io import loadmat
# indsForDaniel = loadmat(output_dir+'indsForDaniel.mat')
# selected =np.array(list(scott_ids.get('selected')))

selected = np.concatenate((np.array(d2c),np.array(c2d)))
# selected = [n-1 for n in selected]
selected_true_ids = conv_1_df.index[selected]

conv_1_all = conv_1_df.loc[selected_true_ids].iloc[:, :1792]
dense_final_all = dense_final_df.loc[selected_true_ids]

corr_layer1_all = conv_1_all.T.corr(method=method)
corr_layer3_all = dense_final_all.T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan).reset_index(drop=True)
corr_layer1_all_triu.columns = range(corr_layer1_all_triu.shape[1])
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan).reset_index(drop=True)
corr_layer3_all_triu.columns = range(corr_layer3_all_triu.shape[1])
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(3))

print('1. correlation between triu of conv1 from c2d and DF from c2d: ', corr_between_layers_c2d.round(3))
print('2. correlation between triu of conv1 from d2c and DF from d2c: ', corr_between_layers_d2c.round(3))
print('3. correlation between triu of conv1 from selected and DF from selected: ', corr_between_layers.round(3))




# Compare my ids vs Scotts

mine = np.concatenate((final_ids_c2d,final_ids_d2c))
len(set(selected_true_ids) - set(mine))
count = 0
for i in selected_true_ids:
    if i not in mine:
        count+=1



























# 2. Clustering d2c
# ==============================================================================
layer = dense_final[:]
n_clusters = 64

file_name = 'd2c'
cluster_labels, sample_silhouette_values = cluster(n_clusters, layer, output_dir, file_name)




# create dictionary {cluster_0: [silh_value_id0, ...],....}

# create dictionary {cluster_0: [id_highest_silh_val,..., lowest_silh],

d_silh = {}
d_ids = {}
for i in range(len(cluster_labels)):
    cluster_label = cluster_labels[i]
    id = ids[i]
    silh_value = sample_silhouette_values[i]
    d_silh.setdefault(cluster_label, []).append(silh_value)
    d_ids.setdefault(cluster_label, []).append(id)

# TODO: measure of how well it matches original labels.



# Rank slih scores and their ids. Take 24 highest.
subset_number_start = 0
subset_number = 30
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

# Check if correct. Compare score of highest id with highest silh score
# Get a sentence
sent = ids_sorted[20]
# Find index in silh_scores
index_sent = cluster_ids.index(sent)
# Get it's silh_score and compare to nth score
print(cluster_silh_scores[index_sent]==cluster_silh_scores_sorted [20])


df_subset_silh = pd.DataFrame(d_highest_silh)
df_subset_ids = pd.DataFrame(d_highest_ids)

ids2 = df_subset_ids.values.T.flatten()
silh2 = df_subset_silh.values.T.flatten()

cluster_df_20 = pd.DataFrame({'ids':ids2, 'silh_values':silh2})
cluster_df_20.to_csv(output_dir+'subset20_d2c.csv')

# TODO: check if mapping is correct between silh and id

df_subset_ids_flatten = df_subset_ids.values.T.flatten()
layer1 = conv_1_df.loc[df_subset_ids_flatten ]
layer3 = dense_final_df.loc[df_subset_ids_flatten ]


'''
Ok. So, after choosing 20 sentences with high silhouette value, we want the 6 that A) correlate the least among each other in layer1 (find lowest values in RSM of layer1) or B) the ones that correlate least with layer1 (take RSM of layer3 minus RSM of layer1 = differences between correlations at both layers, and then take highest values in resulting matrix).
'''

# Choose sentences that corr low on l1.
# ===================================================================================================
import itertools


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


method = 'pearson'


amount_categories = 64

subset = subset_number-subset_number_start


final_sentences = []
for i in range(0,subset*amount_categories, subset ):
    layer1_single_cat = layer1.iloc[:,:1792].iloc[i:i+subset ]
    corr_layer1 = layer1_single_cat.T.corr(method=method)
    corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1,np.nan)
    # Sort values
    df = corr_layer1_triu[:]
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'])
    # # Iterative process to get lowest corr between layers.
    l = []
    amount_sent =8 #TODO leave in 10 for last run. but it doesnt change final correlation
    sentences = list(set(df.iloc[:10, 0:2].values.flatten()))[:amount_sent]
    for subset_sentences in itertools.combinations(sentences, 6):
        l.append(subset_sentences)
    if len(l)<20:
        # print('more')
        l = []
        sentences = list(set(df.iloc[:20, 0:2].values.flatten()))[:amount_sent ]
        for subset_sentences in itertools.combinations(sentences, 6):
            l.append(subset_sentences)
    corr_between_layers_values = []
    # Iterative process to get lowest corr between layers.
    for subset_sentences in l:
        corr_layer1 = layer1.loc[list(subset_sentences)].T.corr(method=method)
        corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1, np.nan)
        corr_layer3 = layer3.loc[list(subset_sentences)].T.corr(method=method)
        corr_layer3_triu = corr_layer3.where(np.triu(np.ones(corr_layer3.shape)).astype(np.bool)).replace(1, np.nan)
        # corr_between_layers1 = corr_layer1.corrwith(corr_layer3).mean()
        corr_between_layers1 = corr_layer1_triu.corrwith(corr_layer3_triu).mean()
        corr_between_layers_values.append(corr_between_layers1)
    index_sentences = find_nearest(corr_between_layers_values, 0)
    print(corr_between_layers_values[index_sentences])
    final_sentences_single_category = l[index_sentences]
    final_sentences.append(final_sentences_single_category)


final_sentences = [n for i in final_sentences for n in i]

corr_layer1_all = layer1.iloc[:,:1792].loc[final_sentences].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))


np.save(output_dir+'final_ids_d2c.npy', final_sentences)



# Clustering from conv to dense_final c2d
# ==============================================================================
layer = conv_1_df.iloc[:,:1792]
n_clusters = 64
output_dir = '/Users/danielmlow/Dropbox/cnn/experiment/stimuli_final/'
file_name = 'c2d'
cluster_labels, sample_silhouette_values = cluster(n_clusters, layer, output_dir, file_name)

# create dictionary {cluster_0: [silh_value_id0, ...],....}

# create dictionary {cluster_0: [id_highest_silh_val,..., lowest_silh],

d_silh = {}
d_ids = {}
for i in range(len(cluster_labels)):
    cluster_label = cluster_labels[i]
    id = ids[i]
    silh_value = sample_silhouette_values[i]
    d_silh.setdefault(cluster_label, []).append(silh_value)
    d_ids.setdefault(cluster_label, []).append(id)

# TODO: measure of how well it matches original labels.
# Rank slih scores and their ids. Take 24 highest.
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


# Check if correct. Compare score of highest id with highest silh score
# Get a sentence
sent = ids_sorted[20]
# Find index in silh_scores
index_sent = cluster_ids.index(sent)
# Get it's silh_score and compare to nth score
print(cluster_silh_scores[index_sent]==cluster_silh_scores_sorted [20])


df_subset_silh = pd.DataFrame(d_highest_silh)
df_subset_ids = pd.DataFrame(d_highest_ids)

ids2 = df_subset_ids.values.T.flatten()
silh2 = df_subset_silh.values.T.flatten()

cluster_df_20 = pd.DataFrame({'ids':ids2, 'silh_values':silh2})
cluster_df_20.to_csv(output_dir+'subset20_'+file_name+'.csv')

# TODO: check if mapping is correct between silh and id

df_subset_ids_flatten = df_subset_ids.values.T.flatten()

layer1 = conv_1_df.loc[df_subset_ids_flatten ]
layer3 = dense_final_df.loc[df_subset_ids_flatten ]


'''
Ok. So, after choosing 20 sentences with high silhouette value, we want the 6 that A) correlate the least among each other in layer1 (find lowest values in RSM of layer1) or B) the ones that correlate least with layer1 (take RSM of layer3 minus RSM of layer1 = differences between correlations at both layers, and then take highest values in resulting matrix).
'''

# Choose sentences that corr low on l3.
# ===================================================================================================
import itertools


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


method = 'pearson'


amount_categories = 64

subset = subset_number-subset_number_start

# layer_1 = layer1.iloc[:subset*amount_categories,:1792] #remove zero rows. nonzerorows=14*128


iterative = True


final_sentences = []
for i in range(0,subset*amount_categories, subset ):
    # layer1_single_cat = layer1.iloc[:,:1792].iloc[i:i+subset ]
    layer3_single_cat = layer3.iloc[i:i + subset]
    corr_layer3 = layer3_single_cat.T.corr(method=method)
    corr_layer3_triu = corr_layer3.where(np.triu(np.ones(corr_layer3.shape)).astype(np.bool)).replace(1,np.nan)
    # Sort values
    df = corr_layer3_triu[:]
    if np.sum(np.sum(df))==0:
        corr_layer3_triu = corr_layer3[:]
        # Sort values
        df = corr_layer3_triu[:]
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'])
    if iterative:
        # # Iterative process to get lowest corr between layers.
        l = []
        amount_sent = 9
        sentences = list(set(df.iloc[:10, 0:2].values.flatten()))[:amount_sent ]
        for subset_sentences in itertools.combinations(sentences, 6):
            l.append(subset_sentences)
        if len(l)<amount_sent :
            print('more')
            l = []
            sentences = list(set(df.iloc[:20, 0:2].values.flatten()))[:amount_sent ]
            for subset_sentences in itertools.combinations(sentences, 6):
                l.append(subset_sentences)

        # Iterative process to get lowest corr between layers.
        corr_between_layers_values = []
        for subset_sentences in l:
            corr_layer1 = layer1.loc[list(subset_sentences)].T.corr(method=method)
            corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1, np.nan)
            corr_layer3 = layer3.loc[list(subset_sentences)].T.corr(method=method)
            corr_layer3_triu = corr_layer3.where(np.triu(np.ones(corr_layer3.shape)).astype(np.bool)).replace(1, np.nan)
            # corr_between_layers1 = corr_layer3.corrwith(corr_layer3).mean()
            corr_between_layers3 = corr_layer1_triu.corrwith(corr_layer3_triu).mean()
            corr_between_layers_values.append(corr_between_layers3)
        index_sentences = find_nearest(corr_between_layers_values, 0)
        print(corr_between_layers_values[index_sentences])
        final_sentences_single_category = l[index_sentences]
        final_sentences.append(final_sentences_single_category)
    else:
        sentences = list(set(df.iloc[:10, 0:2].values.flatten()))[:6]
        final_sentences.append(sentences)


final_sentences = [n for i in final_sentences for n in i]

corr_layer1_all = layer1.iloc[:,:1792].loc[final_sentences].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))

np.save(output_dir+'final_ids_c2d.npy', final_sentences)











# Combine both
final_ids_c2d = np.load(output_dir+'final_ids_c2d.npy')
final_ids_d2c = np.load(output_dir+'final_ids_d2c.npy')

conv_1_all = conv_1_df.loc[np.concatenate((final_ids_c2d,final_ids_d2c))].iloc[:, :1792]
dense_final_all = dense_final_df.loc[np.concatenate((final_ids_c2d,final_ids_d2c))]

corr_layer1_all = conv_1_all.T.corr(method=method)
corr_layer3_all = dense_final_all.T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(4))




pd.concat(final_ids_c2d )

# Random
import random

rand_smpl = [ conv_1_df.index[i] for i in sorted(random.sample(range(len(conv_1_df.index)),768 )) ]

layer3 = layer3.astype(float)

layer3 = dense_final_df.loc[rand_smpl].T.corr(method=method)

corr_layer1_all = conv_1_df.iloc[:,:1792].loc[rand_smpl].T.corr(method=method)
corr_layer3_all = dense_final_df.loc[rand_smpl].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer3_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()










# corr_between_layers_values_all.append(corr_between_layers) =
# index_sentences_all = find_nearest(corr_between_layers_values, 0)
# print(corr_between_layers_values[index_sentences])
# final_sentences_single_category_all = l[index_sentences]
# final_sentences.append(final_sentences_single_category)
#

























# Choose sentences that corr low on l1.
# ===================================================================================================
import itertools

method = 'pearson'

final_sentences_correlations_L1 = []
final_sentences_correlations_L3 = []

subset = subset_number-subset_number_start




layer_1 = layer1.iloc[:,:1792] #remove zero rows. nonzerorows=14*128

for i in range(0,subset*64, subset ):
    layer1_single_cat = layer1.iloc[i:i+subset ]
    corr_layer1 = layer1_single_cat.T.corr(method=method)
    corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1,np.nan)
    # Sort values
    df = corr_layer1_triu[:]
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'])
    # # Iterative process to get lowest corr between layers.
    l = []
    sentences = set(df.iloc[:5, 0:2].values.flatten())
    for subset_sentences in itertools.combinations(sentences, 6):
        l.append(subset_sentences)
    corr_between_layers_values = []
    # Iterative process to get lowest corr between layers.
    for subset_sentences in l:
        corr_layer1 = layer1.iloc[:,:1792].loc[list(subset_sentences)].T.corr(method=method)
        corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1, np.nan)
        corr_layer3 = layer3.loc[list(subset_sentences)].T.corr(method=method)
        corr_layer3_triu = corr_layer3.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1, np.nan)
        corr_between_layers = corr_layer1_triu.corrwith(corr_layer3_triu).mean()
        corr_between_layers_values.append(corr_between_layers)

    index_sentences = find_nearest(corr_between_layers_values, 0)
    l[index_sentences]


    final_sentences_correlations_single_category_L1 = []
    final_sentences_correlations_single_category_L3 = []
    ids_single_category = []
    value_single_category = []
    j=0
    for z in range(30):
        res = df.iloc[z, :]
        row = int(res[0])
        col = int(res[1])
        if (row not in ids_single_category) and (col not in ids_single_category):#TODO: layer1/layer3
            value_single_category.append(res[2])#TODO: layer1/layer3
            ids_single_category.append(row)  # TODO: layer1/layer3
            ids_single_category.append(col)  # TODO: layer1/layer3
            # elif layer == 'layer3':
            #     sentences_layer_category3.append(row)#TODO: layer1/layer3
            #     sentences_layer_category3.append(col)#TODO: layer1/layer3
            j+=1
            if j==3:
                break


    final_sentences_correlations_single_category_L1.append(np.mean(value_single_category).round(2))
    final_sentences_correlations_single_category_L1.append(ids_single_category)
    # mean corr L3
    corr_layer3_single_category = layer3.loc[ids_single_category].T.corr(method=method)
    corr_layer3_single_category_triu = corr_layer3_single_category.where(np.triu(np.ones(corr_layer3_single_category.shape)).astype(np.bool)).replace(1, np.nan)
    mean_correlations_L3 = np.nanmean(corr_layer3_single_category_triu)
    final_sentences_correlations_single_category_L3.append(np.sum(mean_correlations_L3).round(2))
    final_sentences_correlations_single_category_L3.append(ids_single_category)
    # Append to final_all
    final_sentences_correlations_L1.append(final_sentences_correlations_single_category_L1)
    final_sentences_correlations_L3.append(final_sentences_correlations_single_category_L3)

# Results
mean_corr_per_cluster_L1 = np.mean([n[0] for n in final_sentences_correlations_L1])
mean_corr_per_cluster_L3 = np.mean([n[0] for n in final_sentences_correlations_L3])
# corr_between_layers
final_sentences = [n[1] for n in final_sentences_correlations_L1]
final_sentences = [n for i in final_sentences for n in i]

corr_layer1_all = layer1.iloc[:,:1792].loc[final_sentences].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()





print(corr_between_layers.round(2))
print(mean_corr_per_cluster_L1.round(2))
print(mean_corr_per_cluster_L3.round(2))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



import itertools
l=[]
sentences = set(df.iloc[:5, 0:2].values.flatten())
for subset_sentences in itertools.combinations(sentences, 6):
    l.append(subset_sentences)

print(len(l))

corr_between_layers_values = []



i=0
for subset_sentences in l:
    corr_layer1 = layer1.loc[list(subset_sentences)].T.corr(method=method)
    corr_layer1_triu = corr_layer1.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1, np.nan)
    corr_layer3 = layer3.loc[list(subset_sentences)].T.corr(method=method)
    corr_layer3_triu = corr_layer3.where(np.triu(np.ones(corr_layer1.shape)).astype(np.bool)).replace(1,np.nan)
    corr_between_layers = corr_layer1_triu.corrwith(corr_layer3_triu).mean()
    corr_between_layers_values.append(corr_between_layers )
    i+=1

index_sentences = find_nearest(corr_between_layers_values, 0)








selected = scott_ids.get('selected')[0]

final_sentences =selected[:]


corr_layer1_all = layer1.iloc[final_sentences].T.corr(method=method)
corr_layer3_all = layer3.iloc[final_sentences].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(6))

corr_between_layers = corr_layer1_all_triu_c2d.corrwith(corr_layer3_all_triu_d2c).mean()

















# Take the highest difference.
# ====================================================================
# [ [sum_of differences, [ids]],[sum_of differences, [ids]], ...
# [ [sum_of_correlations, [ids]],[sum_of differences, [ids]], ...
final_sentences_differences = []
final_sentences_correlations_L1 = []
final_sentences_correlations_L3 = []

layer_1 = layer1.iloc[:,:1792] #remove zero rows. nonzerorows=14*128

subset = subset_number-subset_number_start
for i in range(0,1536, subset ):
    layer3_single_cat = layer3.iloc[i:i+subset]
    layer1_single_cat = layer1.iloc[i:i+subset ]
    corr_layer3 = layer3_single_cat.T.corr(method=method)
    corr_layer1 = layer1_single_cat.T.corr(method=method)
    layer3_minus_layer1 =corr_layer3-corr_layer1
    layer3_minus_layer1_triu = layer3_minus_layer1.where(np.triu(np.ones(layer3_minus_layer1.shape)).astype(np.bool)).replace(0,np.nan)
    # Sort values
    df = layer3_minus_layer1_triu[:]
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'], ascending=False)
    # Check
    # print(df.iloc[0,2]==corr_layer3.loc[225351,222981]-corr_layer1.loc[225351, 222981])
    #
    final_sentences_differences_single_category = []
    final_sentences_correlations_single_category_L1 = []
    final_sentences_correlations_single_category_L3 = []
    ids_single_category = []
    value_single_category = []
    j=0
    for z in range(100):
        res = df.iloc[z, :]
        row = int(res[0])
        col = int(res[1])
        if (row not in ids_single_category) and (col not in ids_single_category):#TODO: layer1/layer3
            value_single_category.append(res[2])#TODO: layer1/layer3
            ids_single_category.append(row)  # TODO: layer1/layer3
            ids_single_category.append(col)  # TODO: layer1/layer3
            # elif layer == 'layer3':
            #     sentences_layer_category3.append(row)#TODO: layer1/layer3
            #     sentences_layer_category3.append(col)#TODO: layer1/layer3
            j+=1
            if j==3:
                break
    final_sentences_differences_single_category.append(np.mean(value_single_category).round(2))
    final_sentences_differences_single_category.append(ids_single_category)
    method = 'pearson'
    corr_layer1_single_category = layer1.loc[ids_single_category].T.corr(method=method)
    corr_layer1_single_category_triu = corr_layer1_single_category.where(np.triu(np.ones(corr_layer1_single_category.shape)).astype(np.bool)).replace(1, np.nan)
    mean_correlations_L1 = np.nanmean(corr_layer1_single_category_triu)
    final_sentences_correlations_single_category_L1.append(np.sum(mean_correlations_L1).round(2))
    final_sentences_correlations_single_category_L1.append(ids_single_category)
    # mean correlations layer3
    corr_layer3_single_category = layer3.loc[ids_single_category].T.corr(method=method)
    corr_layer3_single_category_triu = corr_layer3_single_category.where(np.triu(np.ones(corr_layer3_single_category.shape)).astype(np.bool)).replace(1, np.nan)
    mean_correlations_L3 = np.nanmean(corr_layer3_single_category_triu)
    final_sentences_correlations_single_category_L3.append(np.sum(mean_correlations_L3).round(2))
    final_sentences_correlations_single_category_L3.append(ids_single_category)
    # append category data to final_sentences
    final_sentences_differences.append(final_sentences_differences_single_category)
    final_sentences_correlations_L1.append(final_sentences_correlations_single_category_L1)
    final_sentences_correlations_L3.append(final_sentences_correlations_single_category_L3)


# Results
mean_corr_per_cluster_L1 = np.mean([n[0] for n in final_sentences_correlations_L1])
mean_corr_per_cluster_L3 = np.mean([n[0] for n in final_sentences_correlations_L3])
# corr_between_layers
final_sentences = [n[1] for n in final_sentences_correlations_L1]
final_sentences = [n for i in final_sentences for n in i]

corr_layer1_all = layer1.loc[final_sentences].T.corr(method=method)
corr_layer3_all = layer3.loc[final_sentences].T.corr(method=method)
corr_layer1_all_triu = corr_layer1_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_layer3_all_triu = corr_layer3_all.where(np.triu(np.ones(corr_layer1_all.shape)).astype(np.bool)).replace(1,np.nan)
corr_between_layers = corr_layer1_all_triu.corrwith(corr_layer3_all_triu).mean()

print(corr_between_layers.round(2), mean_corr_per_cluster_L1.round(2), mean_corr_per_cluster_L3.round(2))




















df = pd.DataFrame({'a': [1, 10, 8, 11, -1],'b': list('abdce'),'c': [1.0, 2.0, np.nan, 3.0, 4.0]})



corr_layer1 = layer1.corr(method='spearman')





from scipy.stats import spearmanr

# Choose sentences that correlate least between layer 1 and layer 3

lines1 = [int(n.strip()) for n in lines1]
layer1_stimuli_layer3 = layer3.ix[lines1]
layer1_stimuli_layer1 = layer1.ix[lines1]
corr1_3 = layer1_stimuli_layer3.T.corr(method='spearman')
corr1_1 = layer1_stimuli_layer1.T.corr(method='spearman')

corr1_3_triu = corr1_3.where(np.triu(np.ones(corr1_3.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu = corr1_1.where(np.triu(np.ones(corr1_1.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu.corrwith(corr1_3_triu).mean()


l=[]
for i in range(64):
    cluster_ids = d_highest_ids.get(i)
    for j in cluster_ids:
        sentence_conv_1 = conv_1_df.loc[j,:]
        sentence_dense_final= dense_final_df.loc[j,:]
        corr_1_3 =spearmanr(sentence_conv_1, sentence_dense_final)




    silh_scores_argsort = np.array(cluster_silh_scores).argsort()[::-1]
    cluster_ids = d_ids.get(i)
    ids_sorted = [ cluster_ids[j] for j in silh_scores_argsort] #this reorders list of sentence_ids from high to low
    cluster_silh_scores_sorted = cluster_silh_scores[:]
    cluster_silh_scores_sorted.sort(reverse=True)
    l.append((cluster_silh_scores_sorted[:subset_number ][-1])) #keep track of lowest values
    d_highest_silh[i]=np.round(cluster_silh_scores_sorted[:subset_number ], 4)
    d_highest_ids[i]=ids_sorted[:subset_number ]



























print(np.max(sample_silhouette_values))
print(np.min(sample_silhouette_values))

a = sample_silhouette_values.argsort()[::-1]
sample_silhouette_values[a[0]] #highest

# X0_flattened = X0.T.values.flatten() #this has the original sentence ID
rank_silh = pd.DataFrame({'sentence_id': a, 'silh_rank':range(len(a))})
rank_silh = rank_silh.sort_values(by=['sentence_id']) #this has the new ID
rank = np.array(rank_silh.iloc[:,1]) #new ID
rank_df = pd.DataFrame(np.reshape(rank, X0.shape)) #new ID

preselect_stimuli = []
for column in range(64):
    category1_rank = rank_df.iloc[:, column] #new ID
    argsort_rank_silh= list(category1_rank.argsort(axis=0)) #index of new ID
    index = range(332)
    list1, list2 = zip(*sorted(zip(argsort_rank_silh, index)))
    preselect = list(X0.ix[list2[:12], column]) #correct ID #TODO: set
    preselect_stimuli.append(preselect)

preselect_stimuli = pd.DataFrame(preselect_stimuli).T #from layer1/layer3 top30 silh score




all_values = []
if layer=='layer1':
    sentences_layer1 = [] #TODO: layer1/layer3
elif layer=='layer3':
    sentences_layer3 = []  # TODO: layer1/layer3
for i in range(len(categories)):
    stimuli = preselect_stimuli.iloc[:,i] #select column for 1 category
    if layer == 'layer1':
        subset_layer3 = layer3.ix[stimuli] #vectors of pool1/dense_final # TODO here you look for opposite layer
        sentences_layer_category1 = []
    elif layer == 'layer3':
        subset_layer1 = layer1.ix[stimuli]  # vectors of pool1/dense_final # TODO here you look for opposite layer
        df0 = subset_layer1.T.corr(method='spearman')
        sentences_layer_category3 = []
    # subset_layer1.index=range(30)
    df = df0.where(np.triu(np.ones(df0.shape)).astype(np.bool)).replace(1,np.nan)
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'])
    values = []
    j=0
    for z in range(100):
        res = df.iloc[z, :]
        row = int(res[0])
        col = int(res[1])
        if (row not in sentences_layer_category1) and (col not in sentences_layer_category1):#TODO: layer1/layer3
            values.append(res[2])#TODO: layer1/layer3
            if layer == 'layer1':
                sentences_layer_category1.append(row)  # TODO: layer1/layer3
                sentences_layer_category1.append(col)  # TODO: layer1/layer3
            elif layer == 'layer3':
                sentences_layer_category3.append(row)#TODO: layer1/layer3
                sentences_layer_category3.append(col)#TODO: layer1/layer3
            j+=1
            if j==3:
                break
    if layer == 'layer1':
        sentences_layer1.append(sentences_layer_category1)#TODO: layer1/layer3
        all_values.append(values)
    elif layer == 'layer3':
        sentences_layer3.append(sentences_layer_category3)  # TODO: layer1/layer3
        all_values.append(values)



with open(output_dir+'high_pool1.txt', 'a+') as f: #TODO layer1/3
    for i in sentences_layer1: #TODO: layer1/3
        for j in i:
            f.write(X_raw[j]+'\n')

with open(output_dir + 'high_pool1_ids.txt', 'a+') as f: #TODO layer1/3
    for i in sentences_layer1:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_pool1_values.txt', 'a+') as f:#TODO layer1/3
    for i in all_values:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_densefinal.txt', 'a+') as f:  # TODO layer1/3
    for i in sentences_layer3:  # TODO: layer1/3
        for j in i:
            f.write(X_raw[j] + '\n')

with open(output_dir + 'high_densefinal_ids.txt', 'a+') as f:  # TODO layer1/3
    for i in sentences_layer3:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_densefinal_values.txt', 'a+') as f:  # TODO layer1/3
    for i in all_values:
        for j in i:
            f.write(str(j) + '\n')





with open(output_dir+'high_pool1_ids.txt', 'r') as f:
    lines1 = f.readlines()

lines1 = [int(n.strip()) for n in lines1]
layer1_stimuli_layer3 = layer3.ix[lines1]
layer1_stimuli_layer1 = layer1.ix[lines1]
corr1_3 = layer1_stimuli_layer3.T.corr(method='spearman')
corr1_1 = layer1_stimuli_layer1.T.corr(method='spearman')

corr1_3_triu = corr1_3.where(np.triu(np.ones(corr1_3.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu = corr1_1.where(np.triu(np.ones(corr1_1.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu.corrwith(corr1_3_triu).mean()

with open(output_dir+'high_densefinal_ids.txt', 'r') as f:
    lines3 = f.readlines()

lines3 = [int(n.strip()) for n in lines3]
layer3_stimuli_layer3 = layer3.ix[lines3]
layer3_stimuli_layer1 = layer1.ix[lines3]
corr3_3 = layer3_stimuli_layer3.T.corr(method='spearman')
corr3_1 = layer3_stimuli_layer1.T.corr(method='spearman')

corr3_3_triu = corr3_3.where(np.triu(np.ones(corr3_3.shape)).astype(np.bool)).replace(1,np.nan)
corr3_1_triu = corr3_1.where(np.triu(np.ones(corr3_1.shape)).astype(np.bool)).replace(1,np.nan)
corr3_1_triu.corrwith(corr3_3_triu).mean()




from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
from scipy import stats



layer = 'layer1'


n = filtered_ids.dropna().shape[0] #max amount without NaNs

X0 = pd.DataFrame() #random sample of filtered sentences
for column in range(len(categories)):
    category1 = filtered_ids.iloc[:,column]
    random_subset = category1.dropna().sample(n).reset_index(drop=True).astype(int)
    X0[categories[column]] = random_subset

X = [] #vectors of those sentence layer dense_final
y = [] #labels
for column in range(len(categories)):
    for row in range(X0.shape[0]):
        if layer == 'layer1':
            vector = layer1.iloc[X0.iloc[row,column],:] #TODO: choose layer3 or layer1 (pool1)
        elif layer == 'layer3':
            vector = layer3.iloc[X0.iloc[row, column], :]  # TODO: choose layer3 or layer1 (pool1)
        X.append(vector)
        y.append(categories[column])

X = np.array(X)
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


range_n_clusters = range(42,80,1)
range_n_clusters = [64]
silh_values = []


clusterer = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
cluster_labels = clusterer.fit_predict(X)
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)
silh_values.append(sample_silhouette_values)


print(np.max(sample_silhouette_values))
print(np.min(sample_silhouette_values))

a = sample_silhouette_values.argsort()[::-1]
sample_silhouette_values[3952] #highest

# X0_flattened = X0.T.values.flatten() #this has the original sentence ID
rank_silh = pd.DataFrame({'sentence_id': a, 'silh_rank':range(len(a))})
rank_silh = rank_silh.sort_values(by=['sentence_id']) #this has the new ID
rank = np.array(rank_silh.iloc[:,1]) #new ID
rank_df = pd.DataFrame(np.reshape(rank, X0.shape)) #new ID

preselect_stimuli = []
for column in range(64):
    category1_rank = rank_df.iloc[:, column] #new ID
    argsort_rank_silh= list(category1_rank.argsort(axis=0)) #index of new ID
    index = range(68)
    list1, list2 = zip(*sorted(zip(argsort_rank_silh, index)))
    preselect = list(X0.ix[list2[:12], column]) #correct ID #TODO: set
    preselect_stimuli.append(preselect)

preselect_stimuli = pd.DataFrame(preselect_stimuli).T #from layer1/layer3 top30 silh score




all_values = []
if layer=='layer1':
    sentences_layer1 = [] #TODO: layer1/layer3
elif layer=='layer3':
    sentences_layer3 = []  # TODO: layer1/layer3
for i in range(len(categories)):
    stimuli = preselect_stimuli.iloc[:,i] #select column for 1 category
    if layer == 'layer1':
        subset_layer3 = layer3.ix[stimuli] #vectors of pool1/dense_final # TODO here you look for opposite layer
        sentences_layer_category1 = []
    elif layer == 'layer3':
        subset_layer1 = layer1.ix[stimuli]  # vectors of pool1/dense_final # TODO here you look for opposite layer
        df0 = subset_layer1.T.corr(method='spearman')
        sentences_layer_category3 = []
    # subset_layer1.index=range(30)
    df = df0.where(np.triu(np.ones(df0.shape)).astype(np.bool)).replace(1,np.nan)
    df = df.stack().reset_index()
    df.columns = ['Row', 'Column', 'Value']
    df = df.sort_values(by=['Value'])
    values = []
    j=0
    for z in range(100):
        res = df.iloc[z, :]
        row = int(res[0])
        col = int(res[1])
        if (row not in sentences_layer_category1) and (col not in sentences_layer_category1):#TODO: layer1/layer3
            values.append(res[2])#TODO: layer1/layer3
            if layer == 'layer1':
                sentences_layer_category1.append(row)  # TODO: layer1/layer3
                sentences_layer_category1.append(col)  # TODO: layer1/layer3
            elif layer == 'layer3':
                sentences_layer_category3.append(row)#TODO: layer1/layer3
                sentences_layer_category3.append(col)#TODO: layer1/layer3
            j+=1
            if j==3:
                break
    if layer == 'layer1':
        sentences_layer1.append(sentences_layer_category1)#TODO: layer1/layer3
        all_values.append(values)
    elif layer == 'layer3':
        sentences_layer3.append(sentences_layer_category3)  # TODO: layer1/layer3
        all_values.append(values)



with open(output_dir+'high_pool1.txt', 'a+') as f: #TODO layer1/3
    for i in sentences_layer1: #TODO: layer1/3
        for j in i:
            f.write(X_raw[j]+'\n')

with open(output_dir + 'high_pool1_ids.txt', 'a+') as f: #TODO layer1/3
    for i in sentences_layer1:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_pool1_values.txt', 'a+') as f:#TODO layer1/3
    for i in all_values:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_densefinal.txt', 'a+') as f:  # TODO layer1/3
    for i in sentences_layer3:  # TODO: layer1/3
        for j in i:
            f.write(X_raw[j] + '\n')

with open(output_dir + 'high_densefinal_ids.txt', 'a+') as f:  # TODO layer1/3
    for i in sentences_layer3:
        for j in i:
            f.write(str(j) + '\n')

with open(output_dir + 'high_densefinal_values.txt', 'a+') as f:  # TODO layer1/3
    for i in all_values:
        for j in i:
            f.write(str(j) + '\n')





with open(output_dir+'high_pool1_ids.txt', 'r') as f:
    lines1 = f.readlines()

lines1 = [int(n.strip()) for n in lines1]
layer1_stimuli_layer3 = layer3.ix[lines1]
layer1_stimuli_layer1 = layer1.ix[lines1]
corr1_3 = layer1_stimuli_layer3.T.corr(method='spearman')
corr1_1 = layer1_stimuli_layer1.T.corr(method='spearman')

corr1_3_triu = corr1_3.where(np.triu(np.ones(corr1_3.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu = corr1_1.where(np.triu(np.ones(corr1_1.shape)).astype(np.bool)).replace(1,np.nan)
corr1_1_triu.corrwith(corr1_3_triu).mean()

with open(output_dir+'high_densefinal_ids.txt', 'r') as f:
    lines3 = f.readlines()

lines3 = [int(n.strip()) for n in lines3]
layer3_stimuli_layer3 = layer3.ix[lines3]
layer3_stimuli_layer1 = layer1.ix[lines3]
corr3_3 = layer3_stimuli_layer3.T.corr(method='spearman')
corr3_1 = layer3_stimuli_layer1.T.corr(method='spearman')

corr3_3_triu = corr3_3.where(np.triu(np.ones(corr3_3.shape)).astype(np.bool)).replace(1,np.nan)
corr3_1_triu = corr3_1.where(np.triu(np.ones(corr3_1.shape)).astype(np.bool)).replace(1,np.nan)
corr3_1_triu.corrwith(corr3_3_triu).mean()

