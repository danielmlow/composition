import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import pandas as pd
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import config

def classification_report_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = list(filter(None, line.split(' ')))
        print(row_data)
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    # avg line
    str_list = lines[-2].split(' ')
    row_data = list(filter(None, str_list)) # fastest
    row = {}
    row['class'] = row_data[0]+row_data[1]+row_data[2]
    row['precision'] = float(row_data[3])
    row['recall'] = float(row_data[4])
    row['f1_score'] = float(row_data[5])
    row['support'] = float(row_data[6])
    report_data.append(row)
    # build final df
    df_report = pd.DataFrame.from_dict(report_data)
    df_latex = df_report.to_latex()
    df_latex = df_latex.replace('\n\\toprule', '')  # erase top rule, mid rule and bottom rule line
    df_latex = df_latex.replace('\n\\midrule', '')  # erase top rule, mid rule and bottom rule line
    df_latex = df_latex.replace('\n\\bottomrule', '')  # erase top rule, mid rule and bottom rule line
    return df_report, df_latex


# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,save_to='cm.eps'):
    rcParams.update({'font.size': 5})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    # plt.savefig(save_to+'/cm.eps', format='eps', dpi=300)
    return


from matplotlib import rc

def learning_curve(history, output_dir):
    '''
    :param history:  model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
    :return: plot
    '''
    # list all data in history
    plt.clf()
    font = {'size': 10}
    results = list(history.keys())
    loss_results = []
    acc_results = []
    for result in results:
        if result.endswith('acc'):
            acc_results.append(result)
        elif result.endswith('loss'):
            loss_results.append(result)
    for result in acc_results:
        list1 = history[result]
        list1.insert(0, 0)
        plt.xticks(range(4))
        plt.plot(list1)
    plt.legend(acc_results, loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # plt.xticks(range(config.epochs + 1))
    plt.savefig(output_dir+'learning_curve_accuracy'+'.eps', format='eps', dpi=100)
    plt.clf()
    for result in loss_results:
        list2 = history[result]
        list2.insert(0, list2[0]+0.5)
        plt.plot(list2)
    plt.legend(loss_results, loc='lower left')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(4))
    # plt.xticks(range(config.epochs+1))
    plt.rc('font', **font)
    plt.savefig(output_dir+'learning_curve_loss'+'.eps', format='eps', dpi=100)
    return

# output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/runs_cluster/cnn23/'
# history = np.load(output_dir+'history_dict.npy').item()


