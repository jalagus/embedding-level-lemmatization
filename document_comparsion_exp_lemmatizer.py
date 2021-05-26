import torch
import fasttext.util
import numpy as np
import pandas as pd
import glob
import json

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader

from uralicNLP import uralicApi

def load_dataset(dataset_folder):
    news_json_files = glob.glob(dataset_folder, recursive=True)

    df_rows = []

    for news_json_file in news_json_files:
        with open(news_json_file, "r", encoding="utf-8") as fp:
            json_data = json.load(fp)

        for current_article in json_data['data']:
            all_text_lines = []
            for e in current_article['content']:
                if e['type'] == 'text':
                    all_text_lines.append(e['text'].strip())
            b_part = all_text_lines[:len(all_text_lines) // 2]
            e_part = all_text_lines[len(all_text_lines) // 2:]

            if len(b_part) > 0:
                df_rows.append({'beginning': " ".join(b_part), 'end': " ".join(e_part)})
    return df_rows  


class YleDataset(Dataset):
    def __init__(self, dataset_folder = "ylenews-fi-2011-2018-src/data/fi/2018/01/*.json", preloaded_dataset=None, shuffle=False, n_samples=-1, offset=0):
        if preloaded_dataset is None:
            df_rows = load_dataset(dataset_folder)
        else:
            df_rows = preloaded_dataset

        if n_samples > 0:
            dataset = pd.DataFrame(df_rows[offset:offset + n_samples])
        else:
            dataset = pd.DataFrame(df_rows)

        if shuffle:
            dataset = dataset.sample(frac = 1) 

        self.dataset = dataset

    def get_vocab(self):
        all_unique_words = list(set(" ".join(self.dataset.values.flatten()).split()))
        return all_unique_words

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor = self.dataset.iloc[idx]['beginning']
        pos = self.dataset.iloc[idx]['end']

        sample = {'beginning': anchor, 'end': pos}

        return sample

def fasttext_encode(docs, ft_model, lemma_models):
    ret = []

    for doc in docs:
        split_doc = doc.split()
        lemmatized = [uralicApi.lemmatize(w, "fin", word_boundaries=False) for w in split_doc]
        for i, e in enumerate(lemmatized):
            if len(e) == 0:
                lemmatized[i].append(split_doc[i])
        lemmatized = [w[0] for w in lemmatized if len(w) > 0]
        enc = torch.Tensor([ft_model[w] for w in lemmatized])
        for lemma_model in lemma_models:
            enc = lemma_model(enc)
        ret.append(enc.mean(0).numpy())

    return ret 

def acc_by_rank(sim_mat, k=10):
    accuracy_by_rank_list = []
    i_mat = sim_mat.copy()
    for i in range(sim_mat.shape[0]):
        accuracy_by_rank_list.append(np.argmax(i_mat, 1) == np.arange(i_mat.shape[0]))
        i_mat[np.arange(len(i_mat)), i_mat.argmax(1)] = -1
        
    accuracy_by_rank_list = np.array(accuracy_by_rank_list).astype(np.int32)

    accuracy_by_rank = list(np.cumsum(accuracy_by_rank_list.sum(1))[:k] / sim_mat.shape[0])
    return accuracy_by_rank

fasttext.util.download_model('fi', if_exists='ignore')
ft_model = fasttext.load_model('cc.fi.300.bin')

all_res_vecs = []

preloaded_d = load_dataset("ylenews-fi-2011-2018-src/data/fi/2018/0[123]/*.json")

for run_id in range(10):
    print(f"Run ID: {run_id}")
    data_offset = 1000 * run_id

    test_dataset = YleDataset(preloaded_dataset=preloaded_d, n_samples=1000, offset=data_offset)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    lemma_model_chains = []
    lemma_model_chains.append(('proper', [lambda x: x]))

    for chain_name, lemma_model_chain in lemma_model_chains:
        with torch.no_grad():
            beg_encs = []
            end_encs = []       

            for sample_batched in test_dataloader:
                beg_enc  = fasttext_encode(sample_batched['beginning'], ft_model, lemma_model_chain)
                end_enc  = fasttext_encode(sample_batched['end'], ft_model, lemma_model_chain)

                beg_encs.append(beg_enc)
                end_encs.append(end_enc)

            beg_encs = np.concatenate(beg_encs)
            end_encs = np.concatenate(end_encs)

            sim_mat = cosine_similarity(beg_encs, end_encs)

            res_vec = acc_by_rank(sim_mat, 10)
            print(run_id, chain_name, res_vec)
            all_res_vecs.append([run_id, chain_name] + res_vec)

    res_df = pd.DataFrame(all_res_vecs, columns=['run_id', 'word_classes'] + list(range(1,11)))
    res_df.to_csv(f"doc_comparison_results_proper_lemma.csv")
