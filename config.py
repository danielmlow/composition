
local_or_cluster = True#if False, use RUG cluster
english_or_italian = True #if True, use English dataset
toy = False #True = fast check which only uses 3 categories and 1 epochs; set to False to run full model with all categories and for 3 epochs
epochs = 3
verbose = 0
batch_size=512
plot_RSA = True
validation_or_test_phase = False #if False, run final test model
save_checkpoints = False #Set to True when you want to find the perfect early stopping.
sequence_length = 38 #max text length

if local_or_cluster:
    if english_or_italian:
        # English
        # English dataset has 144 categories with X sentences obtained only from the abstract
        dataset_dir = '/Users/danielmlow/Dropbox/cnn/data/wikipedia/wiki_144/'
        train_path = dataset_dir+'train/'
        validation_path = dataset_dir+'validate/'
        test_path = dataset_dir+'test/'
        word_embeddings_path = '/Users/danielmlow/Dropbox/cnn/data/glove.6B/'
        save_to = '/Users/danielmlow/Dropbox/cnn/composition/runs/'
        #  Do it once then save in case any category is manually removed from the list
        # categories0 = os.listdir(dataset_dir+'train/')
        # categories0.remove('.DS_Store')
        # categories = [n[:-4] for n in categories0]
        categories = ['AcademicJournal', 'Activity', 'Actor', 'Aircraft', 'Airline', 'Album', 'Amphibian', 'AnatomicalStructure', 'Animal', 'Anime', 'Arachnid', 'Architect', 'ArtificialSatellite', 'ArtistDiscography', 'Artwork', 'Athlete', 'Automobile', 'AutomobileEngine', 'Award', 'Band', 'Bank', 'BasketballPlayer', 'Book', 'Boxer', 'Bridge', 'Building', 'ChemicalCompound', 'ChristianBishop', 'City', 'Cleric', 'CollegeCoach', 'Comic', 'ComicsCharacter', 'ComicsCreator', 'Company', 'Congressman', 'Convention', 'Country', 'Criminal', 'Cyclist', 'Dam', 'Device', 'Diocese', 'Disease', 'Drug', 'Election', 'Engine', 'Enzyme', 'EthnicGroup', 'EurovisionSongContestEntry', 'FictionalCharacter', 'FigureSkater', 'Film', 'Food', 'FootballLeagueSeason', 'FootballMatch', 'Fungus', 'GaelicGamesPlayer', 'GivenName', 'GolfPlayer', 'GolfTournament', 'GovernmentAgency', 'Governor', 'GridironFootballPlayer', 'HistoricBuilding', 'HistoricPlace', 'HockeyTeam', 'HorseRace', 'Hospital', 'Insect', 'Island', 'Journalist', 'Judge', 'Language', 'Legislature', 'Lighthouse', 'Locomotive', 'Magazine', 'Mammal', 'Manga', 'MartialArtist', 'Mayor', 'MemberOfParliament', 'MilitaryStructure', 'Mineral', 'Monarch', 'MountainRange', 'Museum', 'MusicalArtist', 'NationalFootballLeagueSeason', 'Newspaper', 'Noble', 'OfficeHolder', 'OlympicEvent', 'Painter', 'Park', 'Person', 'Place', 'Planet', 'Plant', 'Play', 'PoliticalParty', 'PowerStation', 'President', 'ProtectedArea', 'PublicTransitSystem', 'RaceHorse', 'RacingDriver', 'RailwayLine', 'RecordLabel', 'ReligiousBuilding', 'Reptile', 'Royalty', 'RugbyClub', 'Saint', 'Sales', 'Scientist', 'Settlement', 'ShoppingMall', 'Single', 'Skier', 'SoapCharacter', 'SoccerClubSeason', 'SoccerPlayer', 'SoccerTournament', 'Song', 'Species', 'SportsTeam', 'Stadium', 'Star', 'Station', 'Swimmer', 'TelevisionEpisode', 'TelevisionSeason', 'TelevisionShow', 'TelevisionStation', 'TennisPlayer', 'Town', 'Venue', 'Village', 'VolleyballPlayer', 'Weapon', 'Website', 'Wrestler']
    else:
        # Italian
        dataset = 6000
        dataset_dir = '/Users/danielmlow/Dropbox/cnn/data/wikipedia/full_article_datasets/6_38_full_article_' + str(dataset) + '/'
        train_path = dataset_dir+'train/'
        validation_path = dataset_dir+'validate/'
        test_path = dataset_dir+'test/'
        word_embeddings_path = '/Users/danielmlow/Dropbox/cnn/data/wiki.it/'
        save_to = '/Users/danielmlow/Dropbox/cnn/composition/runs/'
        categories = ['Aircraft', 'Airport', 'Album', 'AmericanFootballPlayer', 'Artwork', 'AthleticsPlayer', 'Automobile',
                  'BasketballPlayer', 'Book', 'Building', 'Cardinal', 'CelestialBody', 'ChemicalCompound',
                  'ChristianBishop', 'Company', 'Country', 'CyclingCompetition', 'Cyclist', 'Decoration', 'Diocese',
                  'Disease', 'Dog', 'EthnicGroup', 'FictionalCharacter', 'Film', 'Game', 'IceHockeyPlayer', 'Island',
                  'Language', 'MetroStation', 'MilitaryConflict', 'MilitaryPerson', 'MilitaryUnit', 'Mineral',
                  'Monarch', 'MotorcycleRider', 'Mountain', 'Museum', 'MusicalArtist', 'MusicGenre', 'Newspaper',
                  'Politician', 'PopulatedPlace', 'RacingDriver', 'RailwayStation', 'ReligiousBuilding', 'River',
                  'Road', 'Saint', 'Ship', 'SoccerPlayer', 'Software', 'Song', 'Species', 'SportFacility',
                  'SportsSeason', 'SportsTeam', 'Swimmer', 'TelevisionSeason', 'TelevisionShow', 'TennisPlayer',
                  'University', 'VideoGame', 'VolleyballPlayer']

#cluster:
# else:
#     dataset_dir = '../../../../data/s3120155/wiki2/full_article_datasets/6_38_full_article_'+str(dataset)+'/'
#     dataset_path = dataset_dir + '/dataset_full_article.npz'
#     train_path = dataset_dir + 'train/'
#     validation_path = dataset_dir + 'validate/'
#     test_path = dataset_dir + 'test/'
#     word_embeddings_path  = '../../../../data/s3120155/word_embeddings/'
#     save_to = '../../../../data/s3120155/runs_cluster/'
#     categories = []