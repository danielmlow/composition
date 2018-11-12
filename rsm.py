import plot_outputs
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
import data_helpers
import seaborn as sns
import matplotlib.pyplot as plt

method='spearman'


'''
conv1 is 36x128=4608. 14 and 15 words: 1792
conv2 is 18x25614 and 15 words: 1280

'''


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

def plot_cluster_map(output_dir, df, column_names, output_file_name='lstm_1'):
    # layer_name is just used to define the output layer name
    df.columns = column_names
    df.index = column_names
    sns.set(font_scale=1)
    cg = sns.heatmap(df, method='ward',cmap="RdBu_r", vmin = -1., vmax=1.0, cbar_kws={"ticks":[-1., -0.5, 0.0, 0.5, 1.0]})
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    cg.savefig(output_dir + 'RSA_ward_'+ output_file_name+ '.eps', format='eps', dpi=100)


def plot_heatmap(output_dir , df_corr, column_names,output_file_name = 'similarity_experiment', with_corr_values=True, ):
    # layer_name is just used to define the output layer name
    plt.clf()
    df_corr.columns = column_names
    df_corr.index = column_names
    sns.set(font_scale=1.2)
    sns.heatmap(df_corr,cmap="RdBu_r", vmin = -1., vmax=1.0, cbar_kws={"ticks":[-1., -0.5, 0.0, 0.5, 1.0]}, annot=with_corr_values)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout(1.8)
    plt.savefig(output_dir + 'RSA_ward_'+ output_file_name+ '.eps', format='eps', dpi=100)


def plot_cluster_map(output_dir , df_corr, column_names,output_file_name = 'similarity_experiment'):
    # layer_name is just used to define the output layer name
    plt.clf()
    df_corr.columns = column_names
    df_corr.index = column_names
    sns.set(font_scale=1)
    cg = sns.clustermap(df_corr, method='ward', cmap="RdBu_r", vmin=-1., vmax=1.0,
                        cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]})
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # plt.tight_layout(1.8)
    cg.savefig(output_dir + 'RSA_ward_'+ output_file_name+ '.eps', format='eps', dpi=100)



def add_diagonal(df_corr):
    for i in range(df_corr.shape[0]):
            df_corr.iloc[i,i] = 6
    return df_corr






categories = config.categories
# Xtest, Ytest = data_helpers.load_data(config.test_path, categories)

input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_sentences_experiment/other/'
output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'
df0 = pd.read_csv(input_dir+'c2d_cleaned.csv')
df1 = pd.read_csv(input_dir+'d2c_cleaned.csv')
stimuli = np.concatenate([df0.ids,df1.ids])
categories = np.concatenate([df0.category_name,df1.category_name])


input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/cnn41_final_eval_all/'
conv1 = pd.DataFrame(np.load(input_dir+'output_layers_whole_dataset_a.npz')['a'][stimuli]).iloc[:,:1792]
conv2 = pd.DataFrame(np.load(input_dir+'output_layers_whole_dataset_c.npz')['a'][stimuli]).iloc[:,:1536]
dense = pd.DataFrame(np.load(input_dir+'output_layers_whole_dataset_f.npz')['a'][stimuli])

conv1_triu = layer_to_flattened_triu(conv1)
conv2_triu = layer_to_flattened_triu(conv2)
dense_triu = layer_to_flattened_triu(dense)


column_names = ['conv1', 'conv2', 'dense']
df_norm = pd.DataFrame([conv1_triu, conv2_triu, dense_triu])
df_norm_corr = df_norm.T.corr(method='pearson')
plot_heatmap(output_dir, df_norm_corr, column_names,output_file_name = 'cnn_rsm_of_rsms_spearman', with_corr_values=True)





# RSM of dense subset
subset = list(range(0,384,32))
layers = [conv1.iloc[subset,:], conv2.iloc[subset,:], dense.iloc[subset,:]]
layer_names = ['conv1', 'conv2', 'dense']
for layer, layer_name in zip(layers, layer_names):
    plot_cluster_map(output_dir, layer.T.corr(method='spearman'), categories[subset],output_file_name = 'cnn_'+layer_name+'_subset_rsm')

a = layer.T.corr(method='pearson')