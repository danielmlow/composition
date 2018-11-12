import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance
from matplotlib import rcParams

from numpy.random import seed
seed(123)

from scipy.stats.mstats import spearmanr
from scipy.stats.mstats import pearsonr


metric = 'euclidean'
method = 'ward'


test = False
compare = False

if not test:
    plt.switch_backend('agg')

# Plot from scratch
def plot_RSA(output_dir, categories, layer= None, layer_name='lstm_3', amount_sent=None):
    RSA = []
    start = amount_sent
    # corr_method = 'Pearson'
    df0= layer[0:amount_sent] #sentences from first category
    df0 = pd.DataFrame.transpose(df0) #result: vector_len x amount_sent, eg, 100x1000
    # Create pair-wise correlation matrix between sentences from first category and second category
    print('making RSA_arr...')
    for cat in range(len(categories)):
        row = []
        for sent in range(start,layer.shape[0],amount_sent): #sentences from second category
            df = layer[sent:sent+amount_sent]
            df = pd.DataFrame.transpose(df)
            df0.columns = df.columns
            df_corr = df.corrwith(df0, axis=0, drop=False) #corr sentences from two categories TODO: Spearman, will make a difference
            # df_mean1 = spearmanr(df.values.flatten(), df0.values.flatten())[0]
            # df_mean2 = pearsonr(np.array(df).flatten('F'), np.array(df0).flatten('F'))[0]
            df_mean = df_corr.mean().mean() #mean of correlations between sentences of two categories.
            row.append(df_mean) # single value
        df0 = layer[start:start+ amount_sent]
        df0 = pd.DataFrame.transpose(df0)
        start +=amount_sent
        RSA.append(row)
    # insert 0s in the begining
    RSA_copy = RSA[:]
    for cat in range(len(categories)):
        zeros = [0] * (cat + 1)
        RSA_copy[cat] = zeros + RSA_copy[cat]
    # Create diagonal by correlating with itself
    RSA_copy2 = RSA_copy[:]
    start = 0
    cat = 0
    for sent in range(start, layer.shape[0], amount_sent):
        sentences = layer[sent:sent + amount_sent]
        df = pd.DataFrame(sentences)
        df = pd.DataFrame.transpose(df)
        df_corr = df.corrwith(df, axis=0) #TODO: Spearman, but won't make a difference
        df_mean = df_corr.mean().mean()
        RSA_copy2[cat][cat] = df_mean
        cat += 1
    print('Done making RSA_arr.')
    RSA_arr = np.array(RSA_copy)
    # copy upper triangle to bottom triangle
    for i in range(len(categories)):
        for j in range(i, len(categories)):
            RSA_arr[j][i] = RSA_arr[i][j]

    df = pd.DataFrame(RSA_arr, columns=categories, index=categories)
    df.to_csv(output_dir +'RSA_arr_' + layer_name, index=False)
    # np.save(output_dir +'RSA_arr_' + layer_name, RSA_arr)
    correlations = pd.DataFrame(RSA_arr[:], columns=categories, index=categories)
    correlations_array = np.asarray(RSA_arr)
    row_linkage = linkage(distance.pdist(correlations_array, metric=metric), method=method,
                                    optimal_ordering=True)
    col_linkage = linkage(distance.pdist(correlations_array.T, metric=metric), method=method,
                                    optimal_ordering=True)
    sns.set(font_scale=0.5)
    cg = sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=col_linkage, cmap="RdBu_r", vmin=-1.,
                        vmax=1., cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.]})
    #    cg = sns.clustermap(df, method='ward', cmap="RdBu_r",vmin = -0.8, vmax=0.8, cbar_kws={"ticks":[-0.8,-0.4,0.0, 0.4, 0.8]})
    #
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    cg.savefig(output_dir + 'RSA_ward_'+ layer_name + '.eps', format='eps', dpi=100)
    rcParams['lines.linewidth'] = 0.7
    plt.figure(figsize=(9, 8))
    dendrogram(row_linkage, orientation='left', labels=np.array(categories),
               leaf_font_size=2)
    plt.savefig(output_dir + 'dendrogram_'+ layer_name + '.eps', format='eps', dpi=100)
    return

# Plot from saved RSM
def plot_rsm(path_to_dir, RSA_arr, categories, layer_name='lstm_1'):
    # layer_name is just used to define the output layer name
    RSA_arr1 =  np.load(path_to_dir+RSA_arr)
    df = pd.DataFrame(RSA_arr1, columns=categories, index=categories)
    sns.set(font_scale=0.5)
    cg = sns.clustermap(df, method='ward',cmap="RdBu_r", vmin = -1., vmax=1.0, cbar_kws={"ticks":[-1., -0.5, 0.0, 0.5, 1.0]})
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    cg.savefig(path_to_dir + 'RSA_ward_'+ layer_name + '.eps', format='eps', dpi=100)




# Usually, I call this script from lstm.py, but if I want to test it, then:
# ========================================================================================================================================
if test:
    categories = ['AcademicJournal', 'Activity', 'Actor', 'Aircraft', 'Airline', 'Album', 'Amphibian',
                  'AnatomicalStructure', 'Animal', 'Anime', 'Arachnid', 'Architect', 'ArtificialSatellite',
                  'ArtistDiscography', 'Artwork', 'Athlete', 'Automobile', 'AutomobileEngine', 'Award', 'Band', 'Bank',
                  'BasketballPlayer', 'Book', 'Boxer', 'Bridge', 'Building', 'ChemicalCompound', 'ChristianBishop',
                  'City', 'Cleric', 'CollegeCoach', 'Comic', 'ComicsCharacter', 'ComicsCreator', 'Company',
                  'Congressman', 'Convention', 'Country', 'Criminal', 'Cyclist', 'Dam', 'Device', 'Diocese', 'Disease',
                  'Drug', 'Election', 'Engine', 'Enzyme', 'EthnicGroup', 'EurovisionSongContestEntry',
                  'FictionalCharacter', 'FigureSkater', 'Film', 'Food', 'FootballLeagueSeason', 'FootballMatch',
                  'Fungus', 'GaelicGamesPlayer', 'GivenName', 'GolfPlayer', 'GolfTournament', 'GovernmentAgency',
                  'Governor', 'GridironFootballPlayer', 'HistoricBuilding', 'HistoricPlace', 'HockeyTeam', 'HorseRace',
                  'Hospital', 'Insect', 'Island', 'Journalist', 'Judge', 'Language', 'Legislature', 'Lighthouse',
                  'Locomotive', 'Magazine', 'Mammal', 'Manga', 'MartialArtist', 'Mayor', 'MemberOfParliament',
                  'MilitaryStructure', 'Mineral', 'Monarch', 'MountainRange', 'Museum', 'MusicalArtist',
                  'NationalFootballLeagueSeason', 'Newspaper', 'Noble', 'OfficeHolder', 'OlympicEvent', 'Painter',
                  'Park', 'Person', 'Place', 'Planet', 'Plant', 'Play', 'PoliticalParty', 'PowerStation', 'President',
                  'ProtectedArea', 'PublicTransitSystem', 'RaceHorse', 'RacingDriver', 'RailwayLine', 'RecordLabel',
                  'ReligiousBuilding', 'Reptile', 'Royalty', 'RugbyClub', 'Saint', 'Sales', 'Scientist', 'Settlement',
                  'ShoppingMall', 'Single', 'Skier', 'SoapCharacter', 'SoccerClubSeason', 'SoccerPlayer',
                  'SoccerTournament', 'Song', 'Species', 'SportsTeam', 'Stadium', 'Star', 'Station', 'Swimmer',
                  'TelevisionEpisode', 'TelevisionSeason', 'TelevisionShow', 'TelevisionStation', 'TennisPlayer',
                  'Town', 'Venue', 'Village', 'VolleyballPlayer', 'Weapon', 'Website', 'Wrestler']
    # Remove general overlapping categories and Enzyme which has anomalous strings.
    categories = [n for n in categories if n not in ('Activity', 'Company', 'Device', 'OfficeHolder', 'Person', 'Venue', 'ArtistDiscography', 'Athlete','Animal', 'Species', 'Enzyme', 'Settlement', 'Place', 'HistoricPlace')]

    path_to_dir = '/Users/danielmlow/Dropbox/cnn/lstm/runs_cluster/runs/18-03-16-17-42/'

    RSA_arr_1 = np.load(path_to_dir+'RSA_arr_lstm_1.npy')
    RSA_arr_2 = np.load(path_to_dir+'RSA_arr_lstm_2.npy')
    RSA_arr_3 = np.load(path_to_dir+'RSA_arr_lstm_3.npy')
    RSA_arr_conv_2 = np.load(path_to_dir + 'RSA_arr_conv_2.npy')
    RSA_arr_dense_3 = np.load(path_to_dir+'RSA_arr_dense_3.npy')


    plot_rsm(path_to_dir, RSA_arr_1, categories, layer_name='lstm_1')
    plot_rsm(path_to_dir, RSA_arr_2, categories, layer_name='lstm_2')
    plot_rsm(path_to_dir, RSA_arr_3, categories, layer_name='lstm_3')
    plot_rsm(path_to_dir,RSA_arr_dense_3, categories, layer_name='dense_3')
    plot_rsm(path_to_dir, RSA_arr_conv_2, categories, layer_name='conv_2')
    plot_rsm(path_to_dir, RSA_arr_conv_2, categories, layer_name='conv_1')



    # Correlate RSMs: This doesn't really make sense, you need to see , sentence to sentence.
    # ==============================================================================================================
    # Pearson
    # res_1_2 = pd.DataFrame(RSA_arr_1).corrwith(pd.DataFrame(RSA_arr_2)).mean()
    # res_1_3 = pd.DataFrame(RSA_arr_1).corrwith(pd.DataFrame(RSA_arr_3)).mean()
    # res_2_3 = pd.DataFrame(RSA_arr_2).corrwith(pd.DataFrame(RSA_arr_3)).mean()
    #
    # # Spearman
    # spr_result_1_2 = pd.DataFrame(RSA_arr_1).apply(lambda col: col.corr(pd.DataFrame(RSA_arr_2).iloc[:,0], method='spearman'), axis=0).mean()
    # spr_result_1_3 = pd.DataFrame(RSA_arr_1).apply(lambda col: col.corr(pd.DataFrame(RSA_arr_3).iloc[:,0], method='spearman'), axis=0).mean()
    # spr_result_2_3 = pd.DataFrame(RSA_arr_2).apply(lambda col: col.corr(pd.DataFrame(RSA_arr_3).iloc[:,0], method='spearman'), axis=0).mean()

    # Plot three heatmaps in 1 figure, doesn't make sense with clustermap.
    # ======================================================================================================
    # fig, axs = plt.subplots(ncols=3)
    # # sns.set(font_scale=0.5)
    # df1 = pd.DataFrame(RSA_arr_1, columns=categories, index=categories)
    # df2 = pd.DataFrame(RSA_arr_2, columns=categories, index=categories)
    # df3 = pd.DataFrame(RSA_arr_3, columns=categories, index=categories)
    # cg1 = sns.heatmap(data=df1, ax=axs[0])
    # cg2 = sns.heatmap(data=df2, ax=axs[1])
    # cg3 = sns.heatmap(data=df3, ax=axs[2])
    # fig.show()


# Compare sentences across layers
# ======================================================================================================
if compare:
    path_to_dir = '/Users/danielmlow/Dropbox/cnn/lstm/runs_cluster/18-04-10-20-28/'
    loaded = np.load(path_to_dir+'output_layers.npz')
    dense_3 = loaded['e']
    # lstm_2 = loaded['b']
    # lstm_3 = loaded['c']
    try:
        dense_4 = loaded['d']
    except:
        pass
    try:
        dense_5 = loaded['e']
    except:
        pass



