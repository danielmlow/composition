from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import data_helpers #TODO: comment sklearn for Scott.
import config_evaluate_sentences
import numpy as np
import datetime
import os

# Parameters
'''
Here you can specify which model/s you want to use. Just set 
models=['lstm0'] #for 1 model
models=['lstm0', 'cnn4', etc. ] for several models 
['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7', 'cnn0', 'cnn1', 'cnn2', 'cnn3']
'''
models = ['lstm7', 'cnn0']
sequence_length = 26
categories = data_helpers.categories()

def load_sentences(load_sentences_from, input_dir):

    # call file 'sentences'
    if load_sentences_from == 'manually':
        # Just write them here
        sentences0 = ['I am a boy.', 'I am a boy', 'I am a girl']
    elif load_sentences_from == 'txt':
        # Load from txt
        with open(input_dir + 'sentences.txt', 'r') as f:
            sentences0 = f.read().splitlines()
    elif load_sentences_from == 'csv':
        # Load from csv.
        sentences0 = np.array(pd.read_csv(input_dir + 'sentences.csv', header=None)).flatten()
    # Preprocess/clean text
    sentences = [data_helpers.clean_str(sentence) for sentence in sentences0]
    # Turn words into IDs based on training data
    Xtrain = list(pd.read_csv(input_dir + 'training_samples.csv').astype(str).iloc[:, 2])  # load training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Xtrain)
    sequences = tokenizer.texts_to_sequences(sentences)
    sentences_encoded = pad_sequences(sequences, maxlen=sequence_length)
    return sentences, sentences_encoded

def get_output(model, layer_name, batch_size=512, Xtest=None, layer_3d=False, hidden_layer=False):
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    layer_output = intermediate_layer_model.predict(Xtest, batch_size=batch_size, verbose=0)
    if layer_3d:
        layer_output = np.reshape(layer_output,(layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]*layer_output.shape[3]))
    if hidden_layer:
        layer_output = layer_output[:,-1,:] # get last time step   4000x26x250 to 4000x250
    #     # layer_output = np.reshape(layer_output,(layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]))
    layer_output = pd.DataFrame(layer_output)
    return layer_output

def evaluate_sentences(model_to_use, loaded_model, sentences_encoded):
    model_name = model_to_use+'model_'
    # Evaluate sentences in each layer; layers depends on model_to_use
    if model_to_use in ['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5', 'lstm6', 'lstm7']:
        # All LSTMS (above) have these three layers
        lstm_1 = get_output(loaded_model, 'lstm_1', hidden_layer=True, Xtest=sentences_encoded)
        lstm_2 = get_output(loaded_model, 'lstm_2', hidden_layer=True, Xtest=sentences_encoded)
        lstm_3 = get_output(loaded_model, 'lstm_3', Xtest=sentences_encoded)
        if model_to_use in ['lstm0', 'lstm1', 'lstm2', 'lstm3', 'lstm5', 'lstm6']:
            # These ones end in a softmax which is the input to the loss function, softmax_layer is useless for correlations I think.
            softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
        elif model_to_use in ['lstm7']:
            # I changed softmax function to sigmoid which is useful for correlations. So one more layer
            sigmoid_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
        elif model_to_use in ['lstm4']:
            # These have to fully connected layers plus a softmax (useless) layer.
            dense_4 = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)
            dense_5 = get_output(loaded_model, 'dense_5', Xtest=sentences_encoded)
            softmax_layer = get_output(loaded_model, 'dense_1', Xtest=sentences_encoded)  # TODO: Check name
    elif model_to_use in ['cnn0', 'cnn3']:
        # These have 4 main layers: pool1, pool2, dense3 and sigmoid_layer (output_layer). We could also compare using conv1 vs. pool1.
        conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded,
                                     layer_3d=True)  # Each sentence is a matrix So we turn into vector for easier correlations
        pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_3d=True)
        conv_2_reshaped = get_output(loaded_model, 'conv_2', Xtest=sentences_encoded, layer_3d=True)
        pool_2_reshaped = get_output(loaded_model, 'pool_2', Xtest=sentences_encoded, layer_3d=True)
        dense_3 = get_output(loaded_model, 'dense_3', Xtest=sentences_encoded)
        sigmoid_layer = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)  # TODO: Check name
    elif model_to_use in ['cnn1', 'cnn2']:
        # these don't have conv2 and pool2. So just: pool1, dense3 and sigmoid.
        conv_1_reshaped = get_output(loaded_model, 'conv_1', Xtest=sentences_encoded, layer_3d=True)
        pool_1_reshaped = get_output(loaded_model, 'pool_1', Xtest=sentences_encoded, layer_3d=True)
        dense_3 = get_output(loaded_model, 'dense_3',
                             Xtest=sentences_encoded)  # this should have been named dense_2 during design
        sigmoid_layer = get_output(loaded_model, 'dense_4', Xtest=sentences_encoded)  # TODO: Check name
    # Save to files.
    try:
        lstm_1.to_csv(output_dir + model_name + 'lstm_1.csv')
        lstm_2.to_csv(output_dir + model_name + 'lstm_2.csv')
        lstm_3.to_csv(output_dir + model_name + 'lstm_3.csv')
    except:
        pass
    try:
        softmax_layer.to_csv(output_dir + model_name + 'softmax_layer.csv')
    except:
        pass
    try:
        sigmoid_layer.to_csv(output_dir + model_name + 'sigmoid_layer.csv')
    except:
        pass
    try:
        dense_3.to_csv(output_dir + model_name + 'dense_3.csv')
    except:
        pass
    try:
        dense_4.to_csv(output_dir + model_name + 'dense_4.csv')
    except:
        pass
    try:
        dense_5.to_csv(output_dir + model_name + 'dense_5.csv')
    except:
        pass
    try:
        conv_1_reshaped.to_csv(output_dir + model_name + 'conv_1.csv')
        pool_1_reshaped.to_csv(output_dir + model_name + 'pool_1.csv')
    except:
        pass
    try:
        conv_2_reshaped.to_csv(output_dir + model_name + 'conv_2.csv')
        pool_2_reshaped.to_csv(output_dir + model_name + 'pool_2.csv')
    except:
        pass
    print('Sentence evaluations have been saved. ')
    return


# ====================================================================================
# Paths
load_sentences_from = config_evaluate_sentences.load_sentences_from
input_dir = config_evaluate_sentences.input_dir
output_dir = config_evaluate_sentences.output_dir
# Save to new directory so each evaluation doesn't overwrite previous
directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
output_dir = output_dir+ directory_name + '/'
os.makedirs(output_dir)

if __name__ == "__main__":
    sentences, sentences_encoded = load_sentences(load_sentences_from, input_dir)
    for model_to_use in models:
        loaded_model = load_model(input_dir+'models/'+model_to_use+'/model.h5')
        loaded_model.compile('sgd','mse')
        evaluate_sentences(model_to_use, loaded_model,sentences_encoded)



