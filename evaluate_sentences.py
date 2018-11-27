'''
Daniel M. Low

Input sentences through .txt .csv or manually in ./evaluated_sentences/ directory
It will return directory inside ./evaluated_sentences/ with vectors for each layer in an .csv  


'''
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import data_helpers
import numpy as np
import datetime
import os
import config

# Paths
load_sentences_from = 'txt'  # Other options: 'csv' 'manually'
input_dir = './evaluated_sentences/'
model_dir = './runs/'
output_dir = './evaluated_sentences/'
verbose = 0

# Parameters
'''
Here you can specify which model/s you want to use. For example, set 
models=['cnn41'] #for 1 model
models=['lstm0', 'cnn4', etc. ] for several models 
['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7', 'cnn0', 'cnn1', 'cnn2', 'cnn3']
'''
models = ['cnn41']
sequence_length = 38
categories = config.categories


def load_sentences(load_sentences_from, input_dir):
    # call file 'sentences'
    if load_sentences_from == 'manually':
        # Just write them below
        sentences = ['I am a boy.', 'I am a boy', 'I am a girl']
    elif load_sentences_from == 'txt':
        # Load from txt
        with open(input_dir + 'sentences.txt', 'r') as f:
            sentences = f.read().splitlines()
    elif load_sentences_from == 'csv':
        # Load from csv.
        sentences = np.array(pd.read_csv(input_dir + 'sentences.csv', header=None)).flatten()
    # Preprocess/clean text
    sentences = [data_helpers.clean_str_new(sentence) for sentence in sentences]
    # Turn words into IDs based on training data
    Xtrain = Xtrain, Ytrain = data_helpers.load_all_data(config.train_path,config.validation_path, categories, shuffle=False) # I changed this so it combines train and test
    Xtest, Ytest = data_helpers.load_data(config.test_path, categories)
    tokenizer = Tokenizer(filters='')  # depending on word embedding, set lower=False.
    tokenizer.fit_on_texts(np.append(np.array(Xtrain), np.array(Xtest)))
    sequences = tokenizer.texts_to_sequences(sentences)
    sentences_encoded = pad_sequences(sequences, maxlen=sequence_length, padding='post')
    return sentences, sentences_encoded

# This needs to be edited to load LSTM
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

def get_output(model, layer_name, batch_size=512, Xtest=None, layer_2d_or_1d='2d'):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = intermediate_layer_model.predict(Xtest, batch_size=batch_size, verbose=verbose)
    if layer_2d_or_1d=='2d':
        layer_output = np.reshape(layer_output,(len(Xtest), int(layer_output.shape[1])*int(layer_output.shape[3])))
    layer_output = pd.DataFrame(layer_output)
    return layer_output

def evaluate_sentences(model_to_use='cnn41', loaded_model='.path/model.h5', sentences_encoded=None):
    model_name = model_to_use+'model_'
    # Evaluate sentences in each layer; layers depends on model_to_use
    if model_to_use in ['lstm18']: #TODO
        print('Error: need to implement latest model')
        return
        # # All LSTMS  have these layers
        # lstm_1 = get_output(loaded_model, 'lstm_1', hidden_layer=True, Xtest=sentences_encoded)
        # lstm_2 = get_output(loaded_model, 'lstm_2', hidden_layer=True, Xtest=sentences_encoded)
        # lstm_3 = get_output(loaded_model, 'lstm_3', Xtest=sentences_encoded)
        # if model_to_use in ['lstm4']:
        #     # These have to fully connected layers plus a softmax (useless) layer.
        #     dense_4 = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)
        #     dense_5 = get_output(loaded_model, 'dense_5', Xtest=sentences_encoded)
        #     softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)
    elif model_to_use in ['cnn41']:
        # These have 4 main layers: conv1, conv2, dense1 and dense_final
        conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded,
                                     layer_2d_or_1d='2d')  # Each sentence is a matrix So we turn into vector for easier correlations
        pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        conv_2_reshaped = get_output(loaded_model, 'conv_2', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        pool_2_reshaped = get_output(loaded_model, 'pool_2', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        dense_1 = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded, layer_2d_or_1d='1d')
        dense_final = get_output(loaded_model, 'dense_final', Xtest=sentences_encoded, layer_2d_or_1d='1d')
        softmax_final = get_output(loaded_model, 'softmax_final', Xtest=sentences_encoded, layer_2d_or_1d='1d')
    # Save to files.
    # try: #TODO
    #     lstm_1.to_csv(output_dir + model_name + 'lstm_1.csv')
    #     lstm_2.to_csv(output_dir + model_name + 'lstm_2.csv')
    #     lstm_3.to_csv(output_dir + model_name + 'lstm_3.csv')
    # except:
    #     pass
    try:
        # Save to new directory so each evaluation doesn't overwrite previous
        directory_name = datetime.datetime.now().strftime(model_to_use + "_%y-%m-%d-%H-%M-%S")
        new_dir = output_dir + directory_name + '/'
        os.makedirs(new_dir)
        conv_1_reshaped.to_csv(new_dir + model_name + 'conv_1_reshaped.csv')
        pool_1_reshaped.to_csv(new_dir + model_name + 'pool_1_reshaped.csv')
        conv_2_reshaped.to_csv(new_dir + model_name + 'conv_2_reshaped.csv')
        pool_2_reshaped.to_csv(new_dir + model_name + 'pool_2_reshaped.csv')
        dense_1.to_csv(new_dir + model_name + 'dense_1.csv')
        dense_final.to_csv(new_dir + model_name + 'dense_final.csv')
        softmax_final.to_csv(new_dir + model_name + 'softmax_final.csv')
    except:
        pass
    print('======================================')
    print('Sentence evaluations have been saved. ')
    return


# ====================================================================================

if __name__ == "__main__":
    sentences, sentences_encoded = load_sentences(load_sentences_from, input_dir)
    for model in models:
        loaded_model = load_model(os.path.join('runs',model,'model.h5'))
        evaluate_sentences(model_to_use = model, loaded_model = loaded_model, sentences_encoded = sentences_encoded)

## Tests
# ========================================================================================================================
# Load the model, and compare to Xtest sentence saved during run.
# 'volò solo come prototipo e non venne mai avviato alla produzione in serie ma rimane l unico idrovolante ad avere superato la velocità del suono'
# evaluate this sentences it should be equal to dense_final[4]
# Xtest[4]
# Xtest_encoded[4]
# sent4_encoded = np.array([[7740,   89,   31, 1856,    4,   36,   61,  344, 3884,   26,  397, 6,   81,   58,  798,   14,  511, 8546,   43,  483, 2021,    3, 481,    9, 2642,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0]])
# loaded_model = load_model(os.path.join('runs',model,'model.h5'))
# dense_final = evaluate_sentences(model, loaded_model,sent4_encoded)
# dense_final == Xtest.iloc[4,:]
# # Worked
