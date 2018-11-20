from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import data_helpers #TODO: comment sklearn for Scott.
import numpy as np
import datetime
import os
import config


# Paths
load_sentences_from = 'txt'  # Other options: 'csv' 'manually'
input_dir = './evaluated_sentences/'
output_dir = './evaluated_sentences/'
verbose = 0

# Parameters
'''
Here you can specify which model/s you want to use. For example, set 
models=['lstm0'] #for 1 model
models=['lstm0', 'cnn4', etc. ] for several models 
['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7', 'cnn0', 'cnn1', 'cnn2', 'cnn3']
'''
models = ['cnn41_new']
sequence_length = 38
categories = config.categories

# Save to new directory so each evaluation doesn't overwrite previous
directory_name = datetime.datetime.now().strftime(models[0]+"_%y-%m-%d-%H-%M-%S")
output_dir = output_dir+ directory_name + '/'
os.makedirs(output_dir)


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






def evaluate_sentences(model_to_use, loaded_model, sentences_encoded):
    model_name = model_to_use+'model_'
    # Evaluate sentences in each layer; layers depends on model_to_use
    if model_to_use in ['lstm18']: #TODO
        print('Error: need to implement latest model')
        return
        # # All LSTMS (above) have these three layers
        # lstm_1 = get_output(loaded_model, 'lstm_1', hidden_layer=True, Xtest=sentences_encoded)
        # lstm_2 = get_output(loaded_model, 'lstm_2', hidden_layer=True, Xtest=sentences_encoded)
        # lstm_3 = get_output(loaded_model, 'lstm_3', Xtest=sentences_encoded)
        # if model_to_use in ['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm5', 'lstm6']:
        #     # These ones end in a softmax which is the input to the loss function, softmax_layer is useless for correlations I think.
        #     softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
        # elif model_to_use in ['lstm7']:
        #     # I changed softmax function to sigmoid which is useful for correlations. So one more layer
        #     sigmoid_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
        # elif model_to_use in ['lstm4']:
        #     # These have to fully connected layers plus a softmax (useless) layer.
        #     dense_4 = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)
        #     dense_5 = get_output(loaded_model, 'dense_5', Xtest=sentences_encoded)
        #     softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
    elif model_to_use in ['cnn41_old', 'cnn41_new']: #TODO rename dir and change here
        # These have 4 main layers: conv1, conv2, dense1 and dense_final
        conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded,
                                     layer_2d_or_1d='2d')  # Each sentence is a matrix So we turn into vector for easier correlations
        pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        conv_2_reshaped = get_output(loaded_model, 'conv_2', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        pool_2_reshaped = get_output(loaded_model, 'pool_2', Xtest=sentences_encoded, layer_2d_or_1d='2d')
        dense_1 = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded, layer_2d_or_1d='1d')
        dense_final = get_output(loaded_model, 'dense_final', Xtest=sentences_encoded, layer_2d_or_1d='1d')
        softmax_final = get_output(loaded_model, 'softmax_final', Xtest=sentences_encoded, layer_2d_or_1d='1d')  # TODO: Check name
    # Save to files.
    # try: #TODO
    #     lstm_1.to_csv(output_dir + model_name + 'lstm_1.csv')
    #     lstm_2.to_csv(output_dir + model_name + 'lstm_2.csv')
    #     lstm_3.to_csv(output_dir + model_name + 'lstm_3.csv')
    # except:
    #     pass
    try:
        conv_1_reshaped.to_csv(output_dir + model_name + 'conv_1_reshaped.csv')
        pool_1_reshaped.to_csv(output_dir + model_name + 'pool_1_reshaped.csv')
        conv_2_reshaped.to_csv(output_dir + model_name + 'conv_2_reshaped.csv')
        pool_2_reshaped.to_csv(output_dir + model_name + 'pool_2_reshaped.csv')
        dense_1.to_csv(output_dir + model_name + 'dense_1.csv')
        dense_final.to_csv(output_dir + model_name + 'dense_final.csv')
        softmax_final.to_csv(output_dir + model_name + 'softmax_final.csv')
    except:
        pass
    print('Sentence evaluations have been saved. ')
    return


# ====================================================================================

if __name__ == "__main__":
    sentences, sentences_encoded = load_sentences(load_sentences_from, input_dir)
    for model in models:
        loaded_model = load_model(os.path.join('runs',model,'model.h5'))
        loaded_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) #see how each model was compiled in cnn41_final.py
        evaluate_sentences(model, loaded_model,sentences_encoded)


# tests
old = './evaluated_sentences/cnn41_old_18-11-20-11-46-41/'
new = './evaluated_sentences/cnn41_new_18-11-20-12-05-21/'
files_old = os.listdir(old)
files_new = os.listdir(new)

for i in range(7):
    old_file = pd.read_csv(old+files_old[i])
    new_file = pd.read_csv(new+files_new[i])
    print('Needs to be True: '+str(old_file == new_file))


# I should run it and then compare to new.
# Also, compare loading model with loading weights.