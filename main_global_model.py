import numpy
import torch.nn as nn
import torch
import numpy as np
import time
import pandas as pd
import pickle
import os
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt

from model import LinearLemmatizerNet, LemmatizerNet, CompressionLemmatizerNet, ComplexLemmatizerNet
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

fasttext.util.download_model('fi', if_exists='ignore')
ft = fasttext.load_model('cc.fi.300.bin')

word_cases = list(data['case'].unique())
word_case_numbers = list(data['number'].unique())


n_test = 1000
n_val = 1000
n_train = 10000
n_epochs = 50
res_table = []

current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for run_id in range(10):
    dataset = data
    
    test_set = dataset.sample(n_test, random_state=42)

    dataset = dataset.drop(test_set.index, errors='ignore')
    
    dataset = dataset.sample(frac=1) # Shuffle data

    train_word_pairs = []
    dev_word_pairs = []

    dev_word_pair_dict = defaultdict(list)

    train_word_set = set()
    test_word_set = set()

    # Add singulars
    for row in dataset.iterrows():
        w_base, w_inf = row[1][['base', 'inflected']]

        if len(train_word_pairs) < n_train:
            train_word_pairs.append([ft[w_base], ft[w_inf]])
            train_word_set.add(w_base)
        elif len(dev_word_pairs) < n_val:
            if w_base not in train_word_set:
                dev_word_pair_dict[row[1]['case']].append([ft[w_base], ft[w_inf]])
                dev_word_pairs.append([ft[w_base], ft[w_inf]])
                test_word_set.add(w_base)
        else:
            break

    torch_train_words = torch.Tensor(train_word_pairs)
    train_dataset = torch.utils.data.TensorDataset(torch_train_words)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    del train_word_pairs

    loss_fns = [
        ('mse', torch.nn.MSELoss())
    ]

    model_fns = [
        ('simple', LemmatizerNet),
    ]

    for model_name, model_fn in model_fns:
        for loss_name, loss_fn in loss_fns:
            for alpha in [1.0]: # np.linspace(0.0, 1.0, 11):
                model = model_fn()

                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

                print(f"Model: {model_name}, run id: {run_id}, alpha: {alpha}, loss: {loss_name}")
                for epoch in range(n_epochs):
                    time_running = time.time()
                    for datarow in train_loader:
                        inputs = datarow[0][:,1,:]
                        labels = datarow[0][:,0,:]

                        preds = model(inputs)
                        idem_preds = model(preds)
                        label_preds = model(labels)
                        
                        loss = alpha * loss_fn(preds, labels)               # Distance to the lemma
                        loss += (1 - alpha) * loss_fn(preds, idem_preds)    # Idempotency regularization
                                
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        
                    with torch.no_grad():
                        for word_case in dev_word_pair_dict.keys():
    
                            torch_dev_words = torch.Tensor(dev_word_pair_dict[word_case])
                            dev_dataset = torch.utils.data.TensorDataset(torch_dev_words)

                            preds = model(torch_dev_words[:,1]).numpy()
                            labels = torch_dev_words[:,0].numpy()
                            originals = model(torch_dev_words[:,0]).numpy()

                            sim_mat = cosine_similarity(preds, labels)
                            idem_mat = cosine_similarity(originals, labels)
                            acc = np.mean(np.argmax(sim_mat, 0) - np.arange(preds.shape[0]) == 0)
                            idem_acc = np.mean(np.argmax(idem_mat, 0) - np.arange(preds.shape[0]) == 0)
                            res_table.append( (loss_name, word_case, run_id, epoch, alpha, model_name, acc, idem_acc) )
                torch.save(model.state_dict(), f"pretrained_models/{current_date_time}_{loss_name}_global_{model_name}_{run_id}_with_{alpha:.1f}_regularization.pym")

                res_df = pd.DataFrame(res_table, columns=['loss_name', 'word_case', 'run_id', 'epoch', 'alpha', 'model_name', 'accuracy', 'idem_acc'])
                res_df.to_csv(f"global_run_results_{current_date_time}.csv", index=None)