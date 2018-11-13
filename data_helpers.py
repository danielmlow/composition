
import string
import numpy as np
import re
import random
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


label_encoder = LabelEncoder()


def clean_str_new(s):
    """
    Adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py for Italian
    """
    s = re.sub(r"\'s", "", s)
    s = re.sub(r"\'ve", "have", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'", " ", s)
    punc = re.compile('[%s]' % re.escape(string.punctuation))
    s = punc.sub('', s)  # removes punctuation, not accents
    DIGITS = re.compile("[0-9]", re.UNICODE)
    s = DIGITS.sub("#", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.lower()
    s = s.strip()
    return s


def remove_double_spaces(s):
    s = re.sub(r"\s{2,}", " ", s)
    return s



s = 'Richard A. Prior, is from 3.2. But'

def clean_str(s):
    """
    Tokenization/s cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    s = re.sub(r"[^\\p{L}\\s]", " ", s)  # This removes accents, which we want.
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s) #This removes accents, which we want.
    s = re.sub(r"\'s", "", s)
    s = re.sub(r"\'ve", "have", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r",", "", s) #s = re.sub(r",", " ,", s)
    s = re.sub(r"!", "", s)
    # s = re.sub(r"\(", "\(", s)
    # s = re.sub(r"\)", "\) ", s)
    s = re.sub(r"\?", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"  ", " ", s)
    return s.strip().lower()



def load_output_layers(path_to_dir):
    '''These are the output_layers of the Xvalidation set, 100 sentences per 130 categories. Good for RSMs'''
    loaded = np.load(path_to_dir+'output_layers.npz')
    layers = []
    layer1= pd.DataFrame(loaded['a'])
    layers.append(layer1)
    layer2 = pd.DataFrame(loaded['b'])
    layers.append(layer2)
    try:
        layer3 = pd.DataFrame(loaded['c'])
        layers.append(layer3)
    except: pass
    try:
        layer4 = pd.DataFrame(loaded['d'])
        layers.append(layer4)
    except: pass
    try:
        layer5 = pd.DataFrame(loaded['e'])
        layers.append(layer5)
    except: pass
    try:
        layer6 = pd.DataFrame(loaded['f'])
        layers.append(layer6)
    except: pass
    try:
        layer7 = pd.DataFrame(loaded['g'])
        layers.append(layer7)
    except: pass
    try:
        layer8 = pd.DataFrame(loaded['h'])
        layers.append(layer8)
    except: pass
    try:
        layer9 = pd.DataFrame(loaded['i'])
        layers.append(layer9)
    except: pass
    return np.array(layers)

import codecs

def load_data(train_path,categories,shuffle=False,one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [clean_str_new(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]


# def load_data_uppercase(train_path,categories,shuffle=False,one_hot=False):
#     labels = []
#     x_text_list = []
#     class_number = 0
#     for filename in categories:
#         with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
#             x_text1 = [line.strip() for line in f]
#             x_text1 = [clean_str_uppercase(sent) for sent in x_text1]
#             x_text_list.append(x_text1)
#         labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
#         class_number += 1
#     print(str(class_number) + ' classes added')
#     x_text = [item for sublist in x_text_list for item in sublist]
#     # one hot vectors for labels
#     labels = [item for sublist in labels for item in sublist]
#     if one_hot:
#         a = np.array(labels)
#         b = np.zeros((len(labels), len(categories)), dtype=int)
#         b[np.arange(len(labels)), a] = 1
#         y = b.tolist()  # to shuffle it
#     else:
#         y = labels[:]
#     if shuffle:
#         combined = list(zip(x_text, y))
#         random.shuffle(combined)
#         x_text[:], y[:] = zip(*combined)
#     # y = np.asarray(y)
#     return [x_text, y]




def load_data_raw(train_path, categories, shuffle=False, one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [remove_double_spaces(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]


def load_whole_dataset_raw(train_path,validation_path, test_path,categories,load_all=True, shuffle=False,one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [remove_double_spaces(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        if load_all:
            with codecs.open(validation_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
                x_text2 = [line.strip() for line in f]
                x_text2 = [remove_double_spaces(sent) for sent in x_text2]
                x_text_list.append(x_text2)
            with codecs.open(test_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
                x_text3 = [line.strip() for line in f]
                x_text3 = [remove_double_spaces(sent) for sent in x_text3]
                x_text_list.append(x_text3)
            labels.append(np.full(len(x_text2), class_number, dtype=np.int).tolist())
            labels.append(np.full(len(x_text3), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]

def load_whole_dataset(train_path,validation_path, test_path,categories,load_all=True, shuffle=False,one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [clean_str_new(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        if load_all:
            with codecs.open(validation_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
                x_text2 = [line.strip() for line in f]
                x_text2 = [clean_str_new(sent) for sent in x_text2]
                x_text_list.append(x_text2)
            with codecs.open(test_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
                x_text3 = [line.strip() for line in f]
                x_text3 = [clean_str_new(sent) for sent in x_text3]
                x_text_list.append(x_text3)
            labels.append(np.full(len(x_text2), class_number, dtype=np.int).tolist())
            labels.append(np.full(len(x_text3), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]




# The preprocessing
def load_all_data(train_path,validation_path, categories,shuffle=False,one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [clean_str_new(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        with codecs.open(validation_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text2 = [line.strip() for line in f]
            x_text2 = [clean_str_new(sent) for sent in x_text2]
            x_text_list.append(x_text2)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        labels.append(np.full(len(x_text2), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]


def load_train_test(train_path,validation_path, test_path, categories,shuffle=False,one_hot=False):
    labels = []
    x_text_list = []
    class_number = 0
    for filename in categories:
        with codecs.open(train_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text1 = [line.strip() for line in f]
            x_text1 = [clean_str_new(sent) for sent in x_text1]
            x_text_list.append(x_text1)
        labels.append(np.full(len(x_text1), class_number, dtype=np.int).tolist())
        with codecs.open(validation_path + filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text2 = [line.strip() for line in f]
            x_text2 = [clean_str_new(sent) for sent in x_text2]
            x_text_list.append(x_text2)
        with codecs.open(test_path+ filename + '.txt', "r", encoding='utf-8', errors='ignore') as f:
            x_text3 = [line.strip() for line in f]
            x_text3 = [clean_str_new(sent) for sent in x_text3]
            x_text_list.append(x_text3)
        labels.append(np.full(len(x_text2), class_number, dtype=np.int).tolist())
        labels.append(np.full(len(x_text3), class_number, dtype=np.int).tolist())
        class_number += 1
    print(str(class_number) + ' classes added')
    x_text = [item for sublist in x_text_list for item in sublist]
    # one hot vectors for labels
    labels = [item for sublist in labels for item in sublist]
    if one_hot:
        a = np.array(labels)
        b = np.zeros((len(labels), len(categories)), dtype=int)
        b[np.arange(len(labels)), a] = 1
        y = b.tolist()  # to shuffle it
    else:
        y = labels[:]
    if shuffle:
        combined = list(zip(x_text, y))
        random.shuffle(combined)
        x_text[:], y[:] = zip(*combined)
    # y = np.asarray(y)
    return [x_text, y]








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
    return sentences0
    # Preprocess/clean text



# def encode1(Ytrain):
#     integer_encoded = label_encoder.fit_transform(Ytrain)
#     # binary encode
#     onehot_encoder = OneHotEncoder(sparse=False)
#     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#     Ytrain_integer = [item for sublist in integer_encoded for item in sublist]
#     Ytrain_encoded = onehot_encoder.fit_transform(integer_encoded)
#     return Ytrain_encoded, Ytrain_integer

def encode_sentences(Xtrain, sentences, sequence_length=26):
    # Turn words into IDs based on training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Xtrain)
    sequences = tokenizer.texts_to_sequences(sentences)
    sentences_encoded = pad_sequences(sequences, maxlen=sequence_length)
    return sentences_encoded


def load_output_layers(path_to_dir):
    layers = []
    loaded = np.load(path_to_dir+'output_layers.npz')
    layers.append(pd.DataFrame(loaded['a']))
    layers.append(pd.DataFrame(loaded['b']))
    layers.append(pd.DataFrame(loaded['c']))
    try: layers.append(pd.DataFrame(loaded['d']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['e']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['f']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['g']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['h']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['i']))
    except: pass
    return layers

def load_output_layers_whole_dataset(path_to_dir):
    layers = []
    loaded = np.load(path_to_dir+'output_layers_whole_dataset.npz')
    layers.append(pd.DataFrame(loaded['a']))
    layers.append(pd.DataFrame(loaded['b']))
    layers.append(pd.DataFrame(loaded['c']))
    try: layers.append(pd.DataFrame(loaded['d']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['e']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['f']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['g']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['h']))
    except: pass
    try: layers.append(pd.DataFrame(loaded['i']))
    except: pass
    return layers


def load_output_layers_whole_dataset_partial(path_to_dir, layer_letter):
    loaded = np.load(path_to_dir+'output_layers_whole_dataset.npz')
    layer = loaded[layer_letter]
    return layer





def categories(train_path, remove=('')):
    files = os.listdir(train_path)
    categories = [n[:-4] for n in files]
    categories = [e for e in categories if e not in remove]
    return categories




def categories_english(toy=False, no_general_categories=True):
    if toy:
        categories = ['Mammal', 'Reptile', 'Book']  # , 'Book', 'Film']
        return categories
    elif not toy:
        categories = ['AcademicJournal', 'Activity', 'Actor', 'Aircraft', 'Airline', 'Album', 'Amphibian',
                      'AnatomicalStructure', 'Animal', 'Anime', 'Arachnid', 'Architect', 'ArtificialSatellite',
                      'ArtistDiscography', 'Artwork', 'Athlete', 'Automobile', 'AutomobileEngine', 'Award',
                      'Band', 'Bank',
                      'BasketballPlayer', 'Book', 'Boxer', 'Bridge', 'Building', 'ChemicalCompound',
                      'ChristianBishop',
                      'City', 'Cleric', 'CollegeCoach', 'Comic', 'ComicsCharacter', 'ComicsCreator', 'Company',
                      'Congressman', 'Convention', 'Country', 'Criminal', 'Cyclist', 'Dam', 'Device', 'Diocese',
                      'Disease',
                      'Drug', 'Election', 'Engine', 'Enzyme', 'EthnicGroup', 'EurovisionSongContestEntry',
                      'FictionalCharacter', 'FigureSkater', 'Film', 'Food', 'FootballLeagueSeason',
                      'FootballMatch',
                      'Fungus', 'GaelicGamesPlayer', 'GivenName', 'GolfPlayer', 'GolfTournament',
                      'GovernmentAgency',
                      'Governor', 'GridironFootballPlayer', 'HistoricBuilding', 'HistoricPlace', 'HockeyTeam',
                      'HorseRace',
                      'Hospital', 'Insect', 'Island', 'Journalist', 'Judge', 'Language', 'Legislature',
                      'Lighthouse',
                      'Locomotive', 'Magazine', 'Mammal', 'Manga', 'MartialArtist', 'Mayor',
                      'MemberOfParliament',
                      'MilitaryStructure', 'Mineral', 'Monarch', 'MountainRange', 'Museum', 'MusicalArtist',
                      'NationalFootballLeagueSeason', 'Newspaper', 'Noble', 'OfficeHolder', 'OlympicEvent',
                      'Painter',
                      'Park', 'Person', 'Place', 'Planet', 'Plant', 'Play', 'PoliticalParty', 'PowerStation',
                      'President',
                      'ProtectedArea', 'PublicTransitSystem', 'RaceHorse', 'RacingDriver', 'RailwayLine',
                      'RecordLabel',
                      'ReligiousBuilding', 'Reptile', 'Royalty', 'RugbyClub', 'Saint', 'Sales', 'Scientist',
                      'Settlement',
                      'ShoppingMall', 'Single', 'Skier', 'SoapCharacter', 'SoccerClubSeason', 'SoccerPlayer',
                      'SoccerTournament', 'Song', 'Species', 'SportsTeam', 'Stadium', 'Star', 'Station',
                      'Swimmer',
                      'TelevisionEpisode', 'TelevisionSeason', 'TelevisionShow', 'TelevisionStation',
                      'TennisPlayer',
                      'Town', 'Venue', 'Village', 'VolleyballPlayer', 'Weapon', 'Website', 'Wrestler']
        # Remove general overlapping categories and Enzyme which has anomalous ss.
        categories = [n for n in categories if n not in (
        'Activity', 'Company', 'Device', 'OfficeHolder', 'Person', 'Venue', 'ArtistDiscography', 'Athlete',
        'Animal', 'Species', 'Enzyme', 'Settlement', 'Place', 'HistoricPlace')]
        return categories

def duplicates(list_of_sentences):
    seen = {}
    dupes = []

    for x in list_of_sentences:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes, seen

import pickle

def save_obj(obj, path_and_filename):
    with open('obj/' + path_and_filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

