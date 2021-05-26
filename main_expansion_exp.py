import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

from model import LinearLemmatizerNet, LemmatizerNet, CompressionLemmatizerNet, ComplexLemmatizerNet

import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity

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

for exp_word_case in word_cases[10:]:
    for run_id in range(1):
        singular_data = data[(data['case'] == exp_word_case) & (data['number'] == 'singular')]
        singular_cur_data = singular_data.sample(frac=1) # Shuffle data
        
        if len(singular_data) < 1:
            break

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

        torch_train_words = torch.Tensor(train_word_pairs)
        train_dataset = torch.utils.data.TensorDataset(torch_train_words)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        del train_word_pairs

        torch_dev_words = torch.Tensor(dev_word_pairs)
        dev_dataset = torch.utils.data.TensorDataset(torch_dev_words)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=128)
        del dev_word_pairs

        loss_fns = [
            ('mse', torch.nn.MSELoss())
        ]

        model_fns = [
            ('linear', LinearLemmatizerNet),
            ('simple', LemmatizerNet)
        ]

        for model_name, model_fn in model_fns:
            for loss_name, loss_fn in loss_fns:
                for alpha in np.linspace(0.0, 1.0, 11):
                    model = model_fn()

                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

                    print(f"Word case: {exp_word_case}, run id: {run_id}, alpha: {alpha}, loss: {loss_name}")
                    for epoch in range(n_epochs):
                        for datarow in train_loader:
                            inputs = datarow[0][:,0,:]
                            labels = datarow[0][:,1,:]

                            preds = model(inputs)
                            idem_preds = model(preds)
                            label_preds = model(labels)
                            
                            loss = alpha * loss_fn(preds, labels)               # Distance to the lemma
                            loss += (1 - alpha) * loss_fn(preds, idem_preds)    # Idempotency regularization
                                    
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                            
                        with torch.no_grad():
                            preds = model(torch_dev_words[:,0]).numpy()
                            labels = torch_dev_words[:,1].numpy()
                            originals = model(torch_dev_words[:,1]).numpy()

                            sim_mat = cosine_similarity(preds, labels)
                            idem_mat = cosine_similarity(originals, labels)
                            acc = np.mean(np.argmax(sim_mat, 0) - np.arange(preds.shape[0]) == 0)
                            idem_acc = np.mean(np.argmax(idem_mat, 0) - np.arange(preds.shape[0]) == 0)
                            res_table.append( (loss_name, exp_word_case, run_id, epoch, alpha, model_name, acc, idem_acc) )
                        
                        torch.save(model.state_dict(), f"pretrained_models/expansion_{loss_name}_{exp_word_case}_{model_name}_{run_id}_with_{alpha:.1f}_regularization.pym")

                    res_df = pd.DataFrame(res_table, columns=['loss_name', 'word_case', 'run_id', 'epoch', 'alpha', 'model_name', 'accuracy', 'idem_acc'])
                    res_df.to_csv(f"expansion_run_results_{current_date_time}.csv", index=None)