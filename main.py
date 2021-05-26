import torch
import numpy as np
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt

from model import LinearLemmatizerNet
from torch.nn import functional as F

import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity

# https://arxiv.org/pdf/1902.00972.pdf

def load_dataset(f_path = "wiktionary-morphology-1.1/inflections_fi_nounadj.csv"):
    pkl_file = f_path.replace(".csv", ".pickle")

    if os.path.isfile(pkl_file):
        with open(pkl_file, "rb") as fp:
            data = pickle.load(fp)
    else:
        data = pd.read_csv(f_path, names=['inflected', 'base', 'type'])

        data[['case', 'number']] = data['type'].str.split(':', expand=True)
        data['case'] = data['case'].str.replace('case=', '')
        data['number'] = data['number'].str.replace('number=', '')
        data.drop(['type'], inplace=True, axis=1)

        with open(pkl_file, "wb") as fp:
            pickle.dump(data, fp)
    return data

data = load_dataset()

n_test = 1000
n_train = 5000
n_epochs = 25

word_cases = list(data['case'].unique())
word_case_numbers = list(data['number'].unique())

fasttext.util.download_model('fi', if_exists='ignore')
ft = fasttext.load_model('cc.fi.300.bin')

cur_data = data[(data['case'] == 'genative') & (data['number'] == 'singular')]

train_word_pairs = []
test_word_pairs = []

for i, row in enumerate(cur_data.iterrows()):
    w_base, w_inf = row[1][['base', 'inflected']]

    if i < n_train:
        train_word_pairs.append([ft[w_base], ft[w_inf]])
    elif i >= n_train and i < (n_train + n_test):
        test_word_pairs.append([ft[w_base], ft[w_inf]])
    else:
        break

torch_train_words = torch.Tensor(train_word_pairs)
train_dataset = torch.utils.data.TensorDataset(torch_train_words)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
del train_word_pairs


torch_test_words = torch.Tensor(test_word_pairs)
test_dataset = torch.utils.data.TensorDataset(torch_test_words)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
del test_word_pairs



res_table = []

loss_fn = lambda x, y: -torch.mean(F.cosine_similarity(x, y, 1))

for alpha in np.linspace(0.0, 1.0, 11):
    model = LinearLemmatizerNet()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    print(f"Alpha {alpha}:")
    for epoch in range(n_epochs):
        p_bar = train_loader
        for datarow in p_bar:
            inputs = datarow[0][:,0,:]
            labels = datarow[0][:,1,:]

            preds = model(inputs)
            idem_preds = model(preds)        

            loss = alpha * loss_fn(preds, labels)               # Distance to the lemma
            loss += (1 - alpha) * loss_fn(preds, idem_preds)    # Idempotency regularization
                    
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            preds = model(torch_test_words[:,0]).numpy()
            labels = torch_test_words[:,1].numpy()
            sim_mat = cosine_similarity(preds, labels)
            acc = np.sum(np.argmax(sim_mat, 0) - np.arange(preds.shape[0]) == 0)
            res_table.append( (epoch, alpha, acc) )

res_df = pd.DataFrame(res_table, columns=['epoch','alpha','accuracy'])

for alpha in res_df['alpha'].unique():
    res_df[res_df['alpha'] == alpha].reset_index()['accuracy'].plot(label=alpha)
plt.legend()

plot_vals = []
for alpha in res_df['alpha'].unique()[2:]:
    plot_vals.append(list(res_df[res_df['alpha'] == alpha].reset_index()['accuracy'])[-1])
plt.plot(plot_vals)
plt.legend()
