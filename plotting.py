import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plot_df = pd.read_csv("run_results_2020-12-18_22-15-52.csv")

plot_df = plot_df[plot_df['epoch'] == 49]
mse_plot_df = plot_df[plot_df['loss_name'] == 'mse']
cos_plot_df = plot_df[plot_df['loss_name'] == 'cos']

mse_mean_df = mse_plot_df[['word_case', 'alpha', 'accuracy']].groupby(['word_case', 'alpha']).mean().unstack(0)
cos_mean_df = cos_plot_df[['word_case', 'alpha', 'accuracy']].groupby(['word_case', 'alpha']).mean().unstack(0)

for word_case in mse_mean_df.columns:
    mse_mean_df[word_case].plot()

mse_mean_df.idxmax()

plot_df = pd.read_csv("2020-12-18_22-12-39_mse_run_results_model.csv")
plot_df = plot_df[plot_df['epoch'] == 49]

mean_df = plot_df[['word_case', 'model_name', 'accuracy']].groupby(['word_case', 'model_name']).mean().unstack(0)
all_mean_df = plot_df[['model_name', 'accuracy']].groupby(['model_name']).mean()

print(mean_df.to_latex(float_format="%.3f"))
print(all_mean_df.to_latex(float_format="%.3f"))

plot_df = pd.read_csv("2020-12-18_22-12-39_mse_run_results_model.csv")
plot_df = plot_df[plot_df['epoch'] == 0]
plot_df = plot_df[plot_df['word_case'] != 'genative']

run_times = plot_df[['model_name', 'run_time']].groupby(['model_name']).mean()

print(all_mean_df.join(run_times).to_latex(float_format="%.3f"))


plot_df = pd.read_csv("run_results_2020-12-19_21-19-35.csv")
plot_df = plot_df[plot_df['epoch'] == 49]
mse_plot_df = plot_df[plot_df['loss_name'] == 'mse']
mse_mean_df = mse_plot_df[['word_case', 'alpha', 'accuracy']].groupby(['word_case', 'alpha']).mean().unstack(0)


## PLOT ALPHA PLOTS ##

global_plot_df = pd.read_csv("global_doc_comparison_results_mse_global_simple.csv")
global_plot_df = global_plot_df[global_plot_df['word_classes'] == 'global']
global_plot_df = global_plot_df.groupby(['word_classes', 'alpha']).mean()
global_plot_df = global_plot_df.reset_index().drop(['Unnamed: 0', 'run_id'], axis=1)

plot_df = pd.read_csv("doc_comparison_results_mse_partitive_simple.csv")
doc_mean_df = plot_df.groupby(['word_classes', 'alpha']).mean()
doc_mean_df = doc_mean_df.reset_index().drop(['Unnamed: 0', 'run_id'], axis=1)

cols = doc_mean_df.columns.tolist()

doc_mean_df['model'] = "simple"
doc_mean_df = doc_mean_df[['model'] + cols]

#print(doc_mean_df.to_latex(index=None, float_format="%.3f"))

plt.rcParams.update({'font.size': 13})

plt.figure(figsize=(7,6))
for word_case in ['genative', 'genative + inessive + partitive', 'genative + partitive', 'genative + inessive + elative + partitive']:
    ss = doc_mean_df[doc_mean_df['word_classes'] == word_case]
    plt.plot(ss['alpha'], ss['1'], label=word_case.replace("genative", "genitive"))

plt.plot(global_plot_df['alpha'], global_plot_df['1'], label="global")

plt.hlines(0.311, 0.1, 1.0, label="lemmatizer", linestyles='dotted')
plt.hlines(0.286, 0.1, 1.0, label="none", linestyles='dashed')

plt.xlabel('alpha')
plt.legend()
