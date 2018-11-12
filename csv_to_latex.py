import pandas as pd
import numpy as np
import plot_outputs
import importlib
import matplotlib.pyplot as plt

def csv_to_tex(input_path, output_path):
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_excel(input_path+'.xlsx', encoding='utf-8-sig')
    df_latex = df.to_latex(index=False)
    with open(output_path+'.tex', 'w') as f:
        f.write(str(df_latex))


# Learning Curve
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_model/'
d = np.load(input_dir+'history_dict.npy')
# copy and pasted it:
d = {'acc': [0.608408203125, 0.732587890625, 0.80271484375, 0.863232421875, 0.90556640625, 0.931591796875, 0.9465787760416666, 0.9570865885416666, 0.9646321614583333, 0.96919921875], 'val_acc': [0.6915755208333333, 0.7134635416666667, 0.7146744791666667, 0.7097135416666667, 0.7052083333333333, 0.6964973958333334, 0.6946744791666667, 0.693125, 0.6910026041666667, 0.6902604166666667], 'val_loss': [1.0920380201935769, 1.0093031644821167, 1.0757258840401966, 1.2028360558549562, 1.4188584667444228, 1.6316908218463262, 1.766864692568779, 2.0251855607827505, 2.060387311379115, 2.1677825838327407], 'loss': [1.3953080095847448, 0.9260162450869878, 0.6677782966693242, 0.45178923646608987, 0.3067782683918873, 0.21935892856369416, 0.16770731837799152, 0.1326430987815062, 0.10937795276443163, 0.09297534093881647]}
input_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'
importlib.reload(plot_outputs)
plot_outputs.learning_curve(d, input_dir)

'''
:param history:  model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
:return: plot
'''
# list all data in history
plt.clf()
history = {'acc': [0.608408203125, 0.732587890625, 0.80271484375, 0.863232421875, 0.90556640625, 0.931591796875, 0.9465787760416666, 0.9570865885416666, 0.9646321614583333, 0.96919921875], 'val_acc': [0.6915755208333333, 0.7134635416666667, 0.7146744791666667, 0.7097135416666667, 0.7052083333333333, 0.6964973958333334, 0.6946744791666667, 0.693125, 0.6910026041666667, 0.6902604166666667], 'val_loss': [1.0920380201935769, 1.0093031644821167, 1.0757258840401966, 1.2028360558549562, 1.4188584667444228, 1.6316908218463262, 1.766864692568779, 2.0251855607827505, 2.060387311379115, 2.1677825838327407], 'loss': [1.3953080095847448, 0.9260162450869878, 0.6677782966693242, 0.45178923646608987, 0.3067782683918873, 0.21935892856369416, 0.16770731837799152, 0.1326430987815062, 0.10937795276443163, 0.09297534093881647]}
output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'


font = {'size': 10}
results = list(d.keys())
loss_results = []
acc_results = []
for result in results:
    if result.endswith('acc'):
        acc_results.append(result)
    elif result.endswith('loss'):
        loss_results.append(result)
for result in acc_results:
    list1 = d[result]
    list1.insert(0, 0)
    plt.plot(list1)
plt.legend(acc_results, loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(11))
plt.tight_layout()
# plt.xticks(range(config.epochs + 1))
plt.savefig(output_dir + 'learning_curve_accuracy' + '.eps', format='eps', dpi=100)

plt.clf()
for result in loss_results:
    list2 = history[result]
    list2.insert(0, list2[0] + 0.5)
    plt.plot(list2)
plt.legend(loss_results, loc='upper left')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xticks(range(11))
# plt.xticks(range(config.epochs+1))
plt.rc('font', **font)








# Exp 2 similarity judgment
input_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/exp2_results'
csv_to_tex(input_path, input_path)

# Exp 3 text categorization: stimuli for appendix
output_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/exp3_stimuli'
csv_to_tex(output_path, output_path)


# Exp 3 text categorization: results classification 4 way
output_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/exp3_results_4_classification'
csv_to_tex(output_path, output_path)

# Exp 3 text categorization: results vs human categorization

output_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/summary'
csv_to_tex(output_path, output_path)

output_path = '/Users/danielmlow/Dropbox/cnn/mturk/semantic_similarity/stimuli/stimuli_for_thesis'
csv_to_tex(output_path, output_path)

output_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/rouge_results'
csv_to_tex(output_path, output_path)




# Example of sentences
# ============================================================
input_dir = '/Users/danielmlow/Dropbox/cnn/experiment/final_sentences_experiment/other/'
df = pd.read_csv(input_dir+'d2c_cleaned.csv')

rand = np.random.randint(192,size=64)
df1 = df.iloc[:,-2:].iloc[rand]
output_dir = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/'
df1.to_excel(output_dir+'exp1_stimuli_examples.xlsx')

output_path = '/Users/danielmlow/Dropbox/cnn/thesis/manuscript/tables_and_figures/exp1_stimuli_examples'
csv_to_tex(output_path, output_path)


