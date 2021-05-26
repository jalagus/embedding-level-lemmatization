import torch
import numpy as np
import time
import pandas as pd
import pickle
import os
from datetime import datetime

from model import LinearLemmatizerNet, LemmatizerNet, CompressionLemmatizerNet, ComplexLemmatizerNet

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

n_test = 1000 // 2
n_val = 1000 // 2
n_train = 10000 // 2
n_epochs = 50
res_table = []

current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

nn_models = [
    ('linear', LinearLemmatizerNet ),
    ('simple_nn', LemmatizerNet ),
    ('compression', CompressionLemmatizerNet ),
    ('complex', ComplexLemmatizerNet )
]

for exp_word_case in word_cases[1:]:
    for run_id in range(10):
        singular_data = data[(data['case'] == exp_word_case) & (data['number'] == 'singular')]
        plural_data = data[(data['case'] == exp_word_case) & (data['number'] == 'plural')]

        if len(singular_data) > 0:
            singular_test_set = singular_data.sample(n_test, random_state=42)
            plural_test_set = plural_data.sample(n_test, random_state=42)
            test_set = pd.concat([singular_test_set, plural_test_set])
        else:
            plural_test_set = plural_data.sample(n_test * 2, random_state=42)
            test_set = plural_test_set

        singular_cur_data = singular_data.drop(test_set.index, errors='ignore')
        plural_cur_data = plural_data.drop(test_set.index, errors='ignore')
        
        singular_cur_data = singular_cur_data.sample(frac=1) # Shuffle data
        plural_cur_data = plural_cur_data.sample(frac=1) # Shuffle data

        train_word_pairs = []
        dev_word_pairs = []

        # Add singulars
        for row in singular_cur_data.iterrows():
            w_base, w_inf = row[1][['base', 'inflected']]

            if len(train_word_pairs) < n_train:
                train_word_pairs.append([ft[w_base], ft[w_inf]])
            elif len(dev_word_pairs) < n_val:
                dev_word_pairs.append([ft[w_base], ft[w_inf]])
            else:
                break

        # Fill with plurals until we reach n_train * 2
        for row in plural_cur_data.iterrows():
            w_base, w_inf = row[1][['base', 'inflected']]

            if len(train_word_pairs) < n_train * 2:
                train_word_pairs.append([ft[w_base], ft[w_inf]])
            elif len(dev_word_pairs) < n_val * 2:
                dev_word_pairs.append([ft[w_base], ft[w_inf]])
            else:
                break

        torch_train_words = torch.Tensor(train_word_pairs)
        train_dataset = torch.utils.data.TensorDataset(torch_train_words)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
        del train_word_pairs

        torch_dev_words = torch.Tensor(dev_word_pairs)
        del dev_word_pairs

        loss_fn = torch.nn.MSELoss()

        for model_name, model_fn in nn_models:
            model = model_fn()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            time_running = time.time()
            
            print(f"Word case: {exp_word_case}, run id: {run_id}, model {model_name}:")
            for epoch in range(n_epochs):
                for datarow in train_loader:
                    inputs = datarow[0][:,1,:]
                    labels = datarow[0][:,0,:]

                    preds = model(inputs)

                    loss = loss_fn(preds, labels)               # Distance to the lemma
                            
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    preds = model(torch_dev_words[:,1]).numpy()
                    labels = torch_dev_words[:,0].numpy()
                    originals = model(torch_dev_words[:,0]).numpy()

                    sim_mat = cosine_similarity(preds, labels)
                    idem_mat = cosine_similarity(originals, labels)
                    acc = np.mean(np.argmax(sim_mat, 0) - np.arange(preds.shape[0]) == 0)
                    idem_acc = np.mean(np.argmax(idem_mat, 0) - np.arange(preds.shape[0]) == 0)
                    res_table.append( (exp_word_case, run_id, epoch, model_name, time.time() - time_running, acc, idem_acc) )
            
            torch.save(model.state_dict(), f"pretrained_models/{current_date_time}_mse_{exp_word_case}_{model_name}_{run_id}.pym")
        
        res_df = pd.DataFrame(res_table, columns=['word_case', 'run_id', 'epoch', 'model_name', 'run_time', 'accuracy', 'idem_acc'])
        res_df.to_csv(f"{current_date_time}_mse_run_results_model.csv", index=None)