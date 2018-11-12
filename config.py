local_or_cluster = True#if False, use RUG cluster
toy = False
epochs = 3

verbose = 0

batch_size=512
plot_RSA = True
validation_or_test_phase = False #if False, run final test model
save_checkpoints = False#Set to True when you want to find the perfect early stopping.
dataset = 6000 #12000
sequence_length = 38

if local_or_cluster:
    # dataset_dir = '/Users/danielmlow/Dropbox/cnn/data/wikipedia/full_article_datasets/full_article_'+str(dataset)+'/' #TODO: uncomment below
    dataset_dir = '/Users/danielmlow/Dropbox/cnn/data/wikipedia/full_article_datasets/6_38_full_article_' + str(dataset) + '/'  #
    dataset_path = dataset_dir+'/dataset_full_article.npz'
    train_path = dataset_dir+'train/'
    validation_path = dataset_dir+'validate/'
    test_path = dataset_dir+'test/'
    word_embeddings_path = '/Users/danielmlow/Dropbox/cnn/data/wiki.it/'
    save_to = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/'

else:
    dataset_dir = '../../../../data/s3120155/wiki2/full_article_datasets/6_38_full_article_'+str(dataset)+'/'
    dataset_path = dataset_dir + '/dataset_full_article.npz'
    train_path = dataset_dir + 'train/'
    validation_path = dataset_dir + 'validate/'
    test_path = dataset_dir + 'test/'
    word_embeddings_path  = '../../../../data/s3120155/word_embeddings/'
    save_to = '../../../../data/s3120155/runs_cluster/'


categories = ['Aircraft', 'Airport', 'Album', 'AmericanFootballPlayer', 'Artwork', 'AthleticsPlayer', 'Automobile', 'BasketballPlayer', 'Book', 'Building', 'Cardinal', 'CelestialBody', 'ChemicalCompound', 'ChristianBishop', 'Company', 'Country', 'CyclingCompetition', 'Cyclist', 'Decoration', 'Diocese', 'Disease', 'Dog', 'EthnicGroup', 'FictionalCharacter', 'Film', 'Game', 'IceHockeyPlayer', 'Island', 'Language', 'MetroStation', 'MilitaryConflict', 'MilitaryPerson', 'MilitaryUnit', 'Mineral', 'Monarch', 'MotorcycleRider', 'Mountain', 'Museum', 'MusicalArtist', 'MusicGenre', 'Newspaper', 'Politician', 'PopulatedPlace', 'RacingDriver', 'RailwayStation', 'ReligiousBuilding', 'River', 'Road', 'Saint', 'Ship', 'SoccerPlayer', 'Software', 'Song', 'Species', 'SportFacility', 'SportsSeason', 'SportsTeam', 'Swimmer', 'TelevisionSeason', 'TelevisionShow', 'TennisPlayer', 'University', 'VideoGame', 'VolleyballPlayer']
categories_wrong = ['MusicalArtist', 'River', 'Island', 'Species', 'Automobile', 'RacingDriver', 'Dog', 'Newspaper', 'ReligiousBuilding', 'Mineral', 'AthleticsPlayer', 'Company', 'Album', 'CyclingCompetition', 'Artwork', 'ChristianBishop', 'Film', 'MetroStation', 'MotorcycleRider', 'Building', 'VolleyballPlayer', 'Aircraft', 'Diocese', 'Road', 'Disease', 'RailwayStation', 'IceHockeyPlayer', 'MusicGenre', 'MilitaryPerson', 'EthnicGroup', 'TennisPlayer', 'AmericanFootballPlayer', 'SportsTeam', 'TelevisionShow', 'Politician', 'BasketballPlayer', 'Cardinal', 'MilitaryConflict', 'Saint', 'Airport', 'Software', 'ChemicalCompound', 'CelestialBody', 'Game', 'MilitaryUnit', 'Museum', 'SportFacility', 'FictionalCharacter', 'Swimmer', 'TelevisionSeason', 'SoccerPlayer', 'Cyclist', 'SportsSeason', 'Book', 'Mountain', 'Decoration', 'Monarch', 'Ship', 'PopulatedPlace', 'University', 'Language', 'Country', 'Song', 'VideoGame']