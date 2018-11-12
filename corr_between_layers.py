import numpy as np
import pandas as pd
import data_helpers
import config
import os
import random

from numpy.random import seed
seed(123)

# Load sentences and output_layers
# ==============================================================================================================
# input_dir = '/Volumes/Maxtor/runs_cluster/lstm7/'
# path_to_dir = '/Users/danielmlow/Dropbox/cnn/lstm/runs_cluster/18-03-07-22-48/'


# Parameters
'''
Here you can specify which model/s you want to use. Just set 
models=['lstm0'] #for 1 model
models=['lstm0', 'cnn4', etc. ] for several models 
['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7', 'cnn0', 'cnn1', 'cnn2', 'cnn3']
'''

# models = ['lstm8','cnn7']
input_dir = config.save_to
output_dir = config.save_to

# models = ['lstm7', 'cnn0']
sequence_length = config.sequence_length
# categories = config.categories

evaluate_model = False
if evaluate_model:
    statistics = ['spearman', 'pearson']
    models = ['cnn21']

n_random_sentences=20 #amount of sentences used to correlate between layers.
top_n_prototypical_sentences = 300 #will be taking 10 random sentences from this n. 500: max lose 15% prototypicality.

# importlib.reload(data_helpers)

# input_dir = '/Volumes/Maxtor/runs_cluster/'
# model_to_use = 'lstm0'
# output_dir = input_dir+model_to_use+'/'
# Xtrain = list(np.load(input_dir+'Xtrain.npz')['a'])# Load Xtrain in order to encode sentences # np.savez_compressed(input_dir+'Xtrain',a=Xtrain)








# Find prototypical sentences
# ==============================================================================================================

# turn Xvalidation into dataframe of all sentences
# sentences = pd.DataFrame(np.zeros((1000,130)))
#
# for x in range(130):
#     for y in range(1000):
#         sentences.iloc[y,x] = Xvalidation[x*1000+y]
#
# np.savez_compressed(path_to_dir+'sentences_df', sentences)


# Find prototypical sentences of categories: the ones that, on average, correlate the most with other sentences of their category.



def prototypical_sentences(statistic, Xvalidation, Xvalidation_raw, output_dir, layer=None, validation_size = None, amount_sent = None, nan='with_zeros_', categories=None):
    print('finding prototypical sentences...')
    # takes about 10 minutes
    df_prototypical = pd.DataFrame(np.zeros(shape=(amount_sent,1)))
    df_prototypical_score = pd.DataFrame(np.zeros(shape=(amount_sent,1)))
    i=0
    for cat_start in range(0,int(len(categories)*amount_sent),amount_sent): #eg, 960 sentences at a time, sentence for 1 category
        # corrcoef = pd.DataFrame(np.corrcoef(layer.iloc[cat_start:cat_start + amount_sent, :]))
        corrcoef = layer.iloc[cat_start:cat_start+amount_sent].T.corr(method=statistic) #correlate each sentence feat vector with eachother for a single category
        mean_corrcoef = corrcoef.mean(axis=0) #axis=0 is ccrrect regarding true sentence ID
        sorted_means = mean_corrcoef.sort_values(ascending=False)
        sorted_index = pd.DataFrame(sorted_means.index, columns=[categories[i]])
        df_prototypical = pd.concat([df_prototypical,sorted_index], axis=1)
        sorted_means_index_reset = pd.DataFrame(sorted_means, columns=[categories[i]]).reset_index(drop=True)
        df_prototypical_score = pd.concat([df_prototypical_score, sorted_means_index_reset], axis=1)
        i += 1
    df_prototypical = df_prototypical.iloc[:,1:] #The index of the highest correlating sentences
    df_prototypical_score = df_prototypical_score.iloc[:,1:] #The average correlation for that sentence with other sentences of its category
    # turn indexes into sentences: TAKES A WHILE
    print('making df_prototypical_sentences.csv ...')
    df_prototypical_sentences = pd.DataFrame(np.zeros((amount_sent, len(categories))),columns=categories)
    i = 0
    for column in range(len(categories)):
        for row in range(amount_sent):
            df_prototypical_sentences.iloc[row, column] = Xvalidation[int(df_prototypical.iloc[row, column])]
        i += 1
    df_prototypical_sentences_raw = pd.DataFrame(np.zeros((amount_sent, len(categories))),columns=categories)
    for column in range(len(categories)):
        for row in range(amount_sent):
            df_prototypical_sentences_raw.iloc[row, column] = Xvalidation_raw[df_prototypical.iloc[row, column]]
    df_prototypical[categories].to_csv(output_dir + 'df_prototypical_'+str(nan)+statistic+'.csv', index=False, header=True, columns=categories)
    df_prototypical_score[categories].to_csv(output_dir + 'df_prototypical_score_'+str(nan)+statistic+'.csv', index=False, header=True, columns=categories)
    df_prototypical_sentences[categories].to_csv(output_dir + 'df_prototypical_sentences_'+str(nan)+statistic+'.csv', index=False, encoding='utf-8-sig', header=True, columns=categories)
    df_prototypical_sentences_raw[categories].to_csv(
        output_dir + 'df_prototypical_sentences_raw_' + str(nan) + statistic + '.csv', index=False, encoding='utf-8-sig',
        header=True, columns=categories)
    np.savez_compressed(output_dir + 'sentences_prototypical_'+str(nan)+statistic, a=df_prototypical[categories], b=df_prototypical_score[categories], c=df_prototypical_sentences[categories])
    return df_prototypical[categories], df_prototypical_score[categories], df_prototypical_sentences[categories]




# Find prototypical sentences
# ============================================================================================================
# You want to stack X (e.g., 10) sentences in one column

def random_n_prototypical_sentences(n, df_prototypical, df_prototypical_sentences, num_categories=None, layer=None, categories=None):
    prototypical_random10_layerX = pd.DataFrame()
    for i in range(num_categories):
        # Take top 200 prototypical sentences (20% for Xvalidation or Xtest)
        prototypical_sentences_1cat = df_prototypical_sentences.iloc[:top_n_prototypical_sentences,i]
        # Insert column with sentence len
        len_sent1 = []
        for sent in prototypical_sentences_1cat:
            len_sent = len(sent.split(' '))
            len_sent1.append(len_sent)
        prototypical_sentences_1cat_len = pd.concat([pd.DataFrame(prototypical_sentences_1cat), pd.DataFrame(len_sent1)],axis=1)
        # filter sentences 8-16
            # take 10 random sentences. If I take the top10, I may get very similar sentences
        prototypical_sentences_1cat_len_14= prototypical_sentences_1cat_len[prototypical_sentences_1cat_len.iloc[:,1]<15] #sentences less than 17
        prototypical_sentences_1cat_len_8_14 = prototypical_sentences_1cat_len_14[prototypical_sentences_1cat_len_14.iloc[:, 1] > 7]  # sentences more than 7
        if prototypical_sentences_1cat_len_8_14.shape[0]<n:
                prototypical_sentences_cat0_random10 = prototypical_sentences_1cat_len.sample(n, random_state=123)
        # take 10 random sentences. If I take the top10, I may get very similar sentences
        else:
            prototypical_sentences_cat0_random10 = prototypical_sentences_1cat_len_8_14.sample(n, random_state=123)
        # use index to find sentence id
        index = prototypical_sentences_cat0_random10.index
        prototypical_random10 = list(df_prototypical.iloc[index,i])
        prototypical_random10_layerX = prototypical_random10_layerX.append(layer.loc[prototypical_random10,:], ignore_index=True)
    return prototypical_random10_layerX
#
# def corr_two_layers(n,layer1, layer2, df_prototypical_sentences):
#     # Obtain vector for 10 random prototypical sentences per category, e.g., 740x200 layer 4
#     prototypical_random10_layer1 = random_n_prototypical_sentences(n,df_prototypical_sentences, layer=layer1)
#     prototypical_random10_layer3 = random_n_prototypical_sentences(n,df_prototypical_sentences,layer=layer2)
#     # Correlate every sentence with every sentence, e.g., 740x740, n
#     corr_matrix_l1 = prototypical_random10_layer1.T.corr()
#     corr_matrix_l3 = prototypical_random10_layer3.T.corr()
#     # Triu to not correlate diagonal.
#     corr_matrix_l1_triu = pd.DataFrame(np.triu(corr_matrix_l1, k=1)).replace(0, np.nan)
#     corr_matrix_l3_triu = pd.DataFrame(np.triu(corr_matrix_l3, k=1)).replace(0, np.nan)
#     #Correlation, # Compare RSM from L1 and L3
#     res = corr_matrix_l1_triu.corrwith(corr_matrix_l3_triu).mean()
#     return res



def corr_layers(n, df_prototypical,layers, df_prototypical_sentences, layer_names, statistic,categories=None):
    corr_matrices = []
    a = np.zeros((len(layers),len(layers)))
    corr_between_layers = pd.DataFrame(a, columns=layer_names, index=layer_names)
    for layer in layers:
        prototypical_random10_layerX= random_n_prototypical_sentences(n, df_prototypical,df_prototypical_sentences, num_categories=len(categories), layer=layer, categories=categories)
        corr_matrix_lX = prototypical_random10_layerX.T.corr(method=statistic)
        corr_matrix_lX_triu = pd.DataFrame(np.triu(corr_matrix_lX, k=1)).replace(0, np.nan)
        corr_matrices.append(corr_matrix_lX_triu)
    for j in range(0,len(layers)):
        for i in range(0,len(layers)):
            res = corr_matrices[j].corrwith(corr_matrices[i]).mean() #TODO: Pearson. correlation between layers.
            # corr_matrices contains a corr matrix for each layer in a list. corr_matrice[0] is the first corr matrix
            # res = corr_matrices[j].apply(lambda col: col.corr(corr_matrices[i], method=statistic), axis=0).mean() # https://stackoverflow.com/questions/45843761/spearman-correlation-with-corrwith-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
            # res = corr_matrices[j].apply(lambda col: col.corr(corr_matrices[i], method=statistic), axis=0)
            corr_between_layers.iloc[j,i] = res
    return corr_between_layers

# Main


# Correlate all layers
def corr_between_layers_method(statistic, models, layers, input_dir, Xvalidation, categories, nan = 'with_zeros_', model_type=None):
    # models just needs to say the name so ['cnn11', 'lstm12'] for line 185. it can't say [18-05-07-20-04-04']
    for model_to_use in models:
        try: os.mkdir(input_dir + model_to_use + '/')
        except: pass
        output_dir = input_dir + model_to_use + '/'
            # if sentences_prototypical!=None:
            #     df_prototypical, df_prototypical_score, df_prototypical_sentences = np.load(sentences_prototypical)
        df_prototypical, df_prototypical_score, df_prototypical_sentences = prototypical_sentences(statistic, Xvalidation,output_dir, layer=layers[-2], validation_size= len(Xvalidation), amount_sent=int(len(Xvalidation)/len(categories)),nan=nan, categories=categories)
        if (model_to_use.startswith('lstm_cnn')):
            layer_names = ['lstm1', 'conv1', 'pool1', 'dense1', 'dense_final']
        elif (model_to_use.startswith('lstm') or model_type.startswith('lstm')):
            if len(layers) == 4:
                layer_names = ['lstm1', 'dense_general', 'lstm2', 'dense3']
            elif len(layers) == 3:
                layer_names = ['lstm1', 'lstm2', 'dense_general']
            elif len(layers) == 5:
                layer_names = ['lstm1', 'lstm2', 'dense1', 'dense2', 'dense_final']
            else:
                layer_names = ['layer_' + str(n) for n in range(1, len(layers) + 1)]
            corr_between_layers1 = corr_layers(n_random_sentences, df_prototypical,layers, df_prototypical_sentences, layer_names, statistic, categories=categories)
        elif (model_to_use.startswith('cnn') or model_type.startswith('cnn')):
            if len(layers) == 6:
                layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'dense_1', 'dense_final']
            elif len(layers) == 5:
                layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'dense_final']
            elif len(layers) == 7:
                layer_names = ['conv1', 'pool1', 'dense1','dense2','dense3','dense4', 'dense_final']
            else:
                layer_names = ['layer_' + str(n) for n in range(1, len(layers) + 1)]
            corr_between_layers1 = corr_layers(n_random_sentences,df_prototypical, layers, df_prototypical_sentences, layer_names,statistic, categories=categories)
        corr_between_layers1.round(2).to_csv(output_dir+'/corr_between_layers_'+str(nan)+statistic+'.csv')
        return corr_between_layers1


if evaluate_model:
    # #Load a specific model
    input_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/'
    categories = config.categories
    Xvalidation, Yvalidation = data_helpers.load_data(config.validation_path, categories)
    for model_to_use in models:
        print(model_to_use+'-------------')
        for statistic1 in statistics:
            print(statistic1+'----')
            models2 = [model_to_use]
            corr_between_layers1 = corr_between_layers_method(statistic1, models2, input_dir, Xvalidation,categories)





# filename = 'cnn20_multitask'
# loaded = np.load(input_dir+filename+'/sentences_prototypical.npz')
# df_prototypical, df_prototypical_score, df_prototypical_sentences = loaded['a'], loaded['b'], loaded['c']
# df_prototypical= pd.read_csv(input_dir+file_name+'/df_prototypical_spearman.csv')
# df_prototypical_sentences= pd.read_csv(input_dir+file_name+'/df_prototypical_sentences_spearman.csv')


# ========================================================================================================
# file_name = 'cnn8.py'
# dir = ['cnn8.py']
# save_to = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/'
# corr_between_layers1 = corr_between_layers_method(file_name, dir, save_to, Xvalidation,
#                                                                       categories)
#
# loaded = np.load('/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/cnn8.py/sentences_prototypical.npz')
#
# df_prototypical_score= pd.DataFrame(loaded['b'], columns=categories)
# df_prototypical_sentences = pd.DataFrame(loaded['c'], columns=categories)
#
# l = ['Wine', 'TennisTournament', 'Asteroid']
# l2 = [n for n in categories if n not in l]
#
# for i in l2:
#     a = df_prototypical_sentences.groupby(i).count()
#     b = [n[:200] for n in a.index]
#     c = list(a.iloc[:, 0])
#     with open('/Users/danielmlow/Dropbox/cnn/thesis/other_sentences_71_categories.txt', 'a+') as f:
#         f.write(i + '--------------------\n')
#         for j in range(len(b)):
#             f.write(str(c[j])+' '+b[j]+'\n')
#         f.write('--------------------\n\n')
#


#
#
# # Just correlate two layers
#
# l = []
# for model_to_use in models:
#     layers = data_helpers.load_output_layers(input_dir+model_to_use+'/')
#     # takes about 3 minutes
#     df_prototypical, df_prototypical_score, df_prototypical_sentences = prototypical_sentences(Xvalidation,output_dir, layer=layers[-1], validation_size= len(Yvalidation), amount_sent=int(len(Yvalidation)/len(categories)))
#     if model_to_use[:4]=='lstm':
#         res = corr_two_layers(10, layers[0], layers[-1], df_prototypical_sentences)
#     elif model_to_use[:3]=='cnn':
#         res = corr_two_layers(10, layers[1], layers[-1], df_prototypical_sentences)
#     print(res)
#     l.append([model_to_use, res])
#
# df = pd.DataFrame({'model': [n[0] for n in l],'corr_l1_l3': [n[1] for n in l]})
#
# for model_to_use in models:
#     df_prototypical_sentences.columns=categories
#     df_prototypical_sentences.to_csv('/Users/danielmlow/Dropbox/cnn/lstm/runs_cluster/cnn0/prototypical_sentences.csv')
#     df_prototypical_score.to_csv('/Users/danielmlow/Dropbox/cnn/lstm/runs_cluster/cnn0/prototypical_scores.csv')
# # Use the same prototypical sentences across models using proto_sents of lstm0, for instance
#
# model_to_use = 'lstm0'
# layers = data_helpers.load_output_layers(input_dir+model_to_use+'/')
# df_prototypical, df_prototypical_score, df_prototypical_sentences = prototypical_sentences(Xvalidation,output_dir, layer=layers[-1])
#
# l = []
# models = ['lstm0', 'lstm1', 'lstm2','cnn0', 'cnn1', 'cnn2', 'cnn3']
# for model_to_use in models:
#     layers = data_helpers.load_output_layers(input_dir + model_to_use + '/')
#     if model_to_use[:4]=='lstm':
#         res = corr_two_layers(10, layers[0], layers[1], df_prototypical_sentences)
#     elif model_to_use[:3]=='cnn':
#         res_1_2 = corr_two_layers(10, layers[1], layers[3], df_prototypical_sentences)
#         res_1_3 = corr_two_layers(10, layers[1], layers[4], df_prototypical_sentences)
#         res_1_4 = corr_two_layers(10, layers[1], layers[5], df_prototypical_sentences)
#         res_2_3 = corr_two_layers(10, layers[3], layers[4], df_prototypical_sentences)
#         res_2_4 = corr_two_layers(10, layers[3], layers[5], df_prototypical_sentences)
#         res_3_4 = corr_two_layers(10, layers[4], layers[5], df_prototypical_sentences)
#     print(res)
#     l.append([model_to_use, res])
#
#
# df2 = pd.DataFrame({'model': [n[0] for n in l],'corr_l1_l3': [n[1] for n in l]})
# df2.to_csv('/Users/danielmlow/Dropbox/cnn/lstm/reports/corr_l1_l3_same_protot_sents_POOL.csv')
#
# df3 = pd.DataFrame({'model': [n[0] for n in l],'corr_l1_l3': [n[1] for n in l]})
# df3.to_csv('/Users/danielmlow/Dropbox/cnn/lstm/reports/corr_l1_l3_same_protot_sents_CONV.csv')
#
# df4 = pd.DataFrame({'model': [n[0] for n in l],'corr_l1_l3': [n[1] for n in l]})
# df4.to_csv('/Users/danielmlow/Dropbox/cnn/lstm/reports/corr_l1_l3_same_protot_sents_l2-l3_POOL.csv')
#
#
#
#
# # ================================================================================================================
#
#
#
# def get_output(model, layer_name, batch_size=512, Xtest=None, layer_3d=False, hidden_layer=False):
#     intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#     layer_output = intermediate_layer_model.predict(Xtest, batch_size=batch_size, verbose=0)
#     if layer_3d:
#         layer_output = np.reshape(layer_output,(layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]*layer_output.shape[3]))
#     if hidden_layer:
#         layer_output = layer_output[:,-1,:] # get last time step   4000x26x250 to 4000x250
#     #     # layer_output = np.reshape(layer_output,(layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]))
#     layer_output = pd.DataFrame(layer_output)
#     return layer_output
#
#
#
# def evaluate_sentences(model_to_use, loaded_model, sentences_encoded=None):
#     model_name = model_to_use+'model_'
#     # Evaluate sentences in each layer; layers depends on model_to_use
#     if model_to_use in ['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7']:
#         # All LSTMS (above) have these three layers
#         lstm_1 = get_output(loaded_model, 'lstm_1', hidden_layer=True, Xtest=sentences_encoded)
#         lstm_2 = get_output(loaded_model, 'lstm_2', hidden_layer=True, Xtest=sentences_encoded)
#         lstm_3 = get_output(loaded_model, 'lstm_3', Xtest=sentences_encoded)
#         if model_to_use in ['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm5', 'lstm6']:
#             # These ones end in a softmax which is the input to the loss function, softmax_layer is useless for correlations I think.
#             softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
#             return lstm_1, lstm_2, lstm_3
#         elif model_to_use in ['lstm7']:
#             # I changed softmax function to sigmoid which is useful for correlations. So one more layer
#             sigmoid_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
#             return lstm_1, lstm_2, lstm_3, sigmoid_layer
#         elif model_to_use in ['lstm4']:
#             # These have to fully connected layers plus a softmax (useless) layer.
#             dense_4 = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)
#             dense_5 = get_output(loaded_model, 'dense_5', Xtest=sentences_encoded)
#             softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
#             return lstm_1, lstm_2, lstm_3, dense_4, dense_5
#     elif model_to_use in ['cnn0', 'cnn3']:
#         # These have 4 main layers: pool1, pool2, dense3 and sigmoid_layer (output_layer). We could also compare using conv1 vs. pool1.
#         conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded,
#                                      layer_3d=True)  # Each sentence is a matrix So we turn into vector for easier correlations
#         pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_3d=True)
#         conv_2_reshaped = get_output(loaded_model, 'conv_2', Xtest=sentences_encoded, layer_3d=True)
#         pool_2_reshaped = get_output(loaded_model, 'pool_2', Xtest=sentences_encoded, layer_3d=True)
#         dense_3 = get_output(loaded_model, 'dense_3', Xtest=sentences_encoded)
#         sigmoid_layer = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)  # TODO: Check name
#         return conv_1_reshaped, pool_1_reshaped, conv_2_reshaped, pool_2_reshaped, dense_3, sigmoid_layer
#     elif model_to_use in ['cnn1', 'cnn2']:
#         # these don't have conv2 and pool2. So just: pool1, dense3 and sigmoid.
#         conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded, layer_3d=True)
#         pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_3d=True)
#         dense_3 = get_output(loaded_model, 'dense_3',
#                              Xtest=sentences_encoded)  # this should have been named dense_2 during design
#         sigmoid_layer = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)  # TODO: Check name
#         return conv_1_reshaped, pool_1_reshaped, dense_3, sigmoid_layer
#
# def retrieve_output_layers(input_dir, model_to_use, loaded_model, Xvalidation_encoded):
#     d = {'output_layers_0': [], 'output_layers_1': [], 'output_layers_2': [], 'output_layers_3': [],
#          'output_layers_4': [], 'output_layers_5': []}
#     for batch_range in range(0,Xvalidation_encoded.shape[0], 1000):
#         layers_temp = evaluate_sentences(model_to_use, loaded_model,
#                                          sentences_encoded=Xvalidation_encoded.iloc[batch_range:batch_range + 1000,:])
#         for i in range(len(layers_temp)):
#             d.setdefault('output_layers_' + str(i), []).append(layers_temp[i])
#         print(batch_range)
#     # a = d.get('output_layers_0')
#     layers = []
#     for i in range(len(d.keys())):
#         try:
#             layer = pd.concat(d.get('output_layers_' + str(i)))
#             layers.append(layer)
#         except:
#             pass
#     try:
#         np.savez_compressed(input_dir + model_to_use + '/output_layers.npz', a=layers[0], b=layers[1], c=layers[2],
#                             d=layers[3], e=layers[4], f=layers[5], g=layers[6])
#         print('has 7 layers')
#     except:
#         try:
#             np.savez_compressed(input_dir + model_to_use + '/output_layers.npz', a=layers[0], b=layers[1],
#                                 c=layers[2], d=layers[3], e=layers[4], f=layers[5])
#             print('has 6 layers')
#         except:
#             try:
#                 np.savez_compressed(input_dir + model_to_use + '/output_layers.npz', a=layers[0], b=layers[1],
#                                     c=layers[2], d=layers[3], e=layers[4])
#                 print('has 5 layers')
#             except:
#                 try:
#                     np.savez_compressed(input_dir + model_to_use + '/output_layers.npz', a=layers[0], b=layers[1],
#                                         c=layers[2], d=layers[3])
#                     print('has 4 layers')
#                 except:
#                     try:
#                         np.savez_compressed(input_dir + model_to_use + '/output_layers.npz', a=layers[0],
#                                             b=layers[1], c=layers[2])
#                         print('has 3 layers')
#                     except:
#                         try:
#                             np.savez_compressed(input_dir + 'output_layers_temp.npz', layers)
#                             print('had to save the layers together')
#                         except:
#                             pass
#     return layers
#
#
# def load_output_layers(path_to_dir):
#     '''These are the output_layers of the Xvalidation set, 100 sentences per 130 categories. Good for RSMs'''
#     loaded = np.load(path_to_dir+'output_layers.npz')
#     layers = []
#     layer1= pd.DataFrame(loaded['a'])
#     layers.append(layer1)
#     layer2 = pd.DataFrame(loaded['b'])
#     layers.append(layer2)
#     layer3 = pd.DataFrame(loaded['c'])
#     layers.append(layer3)
#     try:
#         layer4 = pd.DataFrame(loaded['d'])
#         layers.append(layer4)
#     except: pass
#     try:
#         layer5 = pd.DataFrame(loaded['e'])
#         layers.append(layer5)
#     except: pass
#     return np.array(layers)
#
#
# #====
#
#
#
# input_dir = '/Volumes/Maxtor/runs_cluster/'
# model_to_use = 'lstm7'
#
#
# # Load Xvalidation and encode
# # Xtrain = list(np.load(input_dir+'Xtrain.npz')['a'])# Load Xtrain in order to encode sentences # np.savez_compressed(input_dir+'Xtrain',a=Xtrain)
# # prototypical_sentences = ['It can jump 20 feet', 'it catches its preys in the air', 'the building was built in 1920']
# # clean sentences
# # sentences = [data_helpers.clean_str(sentence) for sentence in prototypical_sentences]
# # sentences_encoded = data_helpers.encode_sentences(Xtrain,Xvalidation,sequence_length=sequence_length)
# Xvalidation_encoded = pd.DataFrame(np.load(input_dir+'Xvalidation_encoded.npz')['a'])
#
# # Load model
#
# loaded_model = load_model(input_dir+model_to_use+'/model.h5')
#
# outputlayers_exists = False
#
# layers_2 = data_helpers.load_output_layers(input_dir+model_to_use+'/')
#
#
# if outputlayers_exists:
#     layers= load_output_layers(input_dir+model_to_use+'/')
#
# l = []
# models = ['lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7']  # missing output_layers.npz
# for model_to_use in models:
#     print(model_to_use+'--------------------')
#     loaded_model = load_model(input_dir + model_to_use + '/model.h5')
#     loaded_model.compile('sgd', 'mse')
#     layers = retrieve_output_layers(input_dir, model_to_use,loaded_model,Xvalidation_encoded)
#     l.append(layers)
#
# # rsa.plot_RSA(input_dir+model_to_use+'/', categories, layer=layers[3], layer_name='lstm_1')
#
#
#
#
#
#
