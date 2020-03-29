import pandas as pd
import json
from tqdm.notebook import tqdm
from itertools import product

import os
import numpy as np
import pickle as pkl
from gym import spaces
from scipy.optimize import minimize
from tqdm import tnrange

import sys
from RLMultilayer.utils import visualize_progress, summarize_res, combine_tracker, load_exp_res, DesignTracker, cal_merit_mse, cal_reward
from RLMultilayer.taskenvs.tasks import get_env_fn
import glob

from torch import nn
import torch
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from RLMultilayer.taskenvs import tasks
from RLMultilayer.utils import cal_reward
from RLMultilayer.utils import TMM_sim

import seaborn as sns
sns.set(font_scale=1)
import re 

def finetune(simulator, m0, x0, target, display=False, bounds=None):
    '''
    Finetune the structure using quasi-Newton's method.
    
    Args:
        m0: materials list given by the upstream RL
        x0: thicknesses given by the upstream RL
        display: if true, then plot the spectrum before and after the finetuning.
        
    Returns:
        x_opt: finetuned thickness list
    '''
    
    def objective_func(x):
        R, T, A = simulator.spectrum(m0, [np.inf]+list(x)+[np.inf])
        return 1-cal_reward(R, T, A, target)
    
    if bounds is None:
        bounds = [(5, 200)] * len(x0)

    print('Initial reward {}'.format(1-objective_func(x0)))
    res = minimize(objective_func, x0, bounds=bounds, options={'disp':True})
    x_opt = [int(item) for item in res.x]
    
    if display:
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x0+[np.inf], title=True, plot=True)
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x_opt+[np.inf], title=True, plot=True)
    
    return x_opt, res

def plot_results(folder, col=None, row=None, hue=None):
    
    df = load_exp_res(folder)
    sns.set(font_scale=1)
    
    reward_factor = df['reward_factor']
    df['LowEpRet'] = (df['AverageEpRet'] - 0.5 * df['StdEpRet']) / reward_factor
    df['HighEpRet'] = (df['AverageEpRet'] + 0.5 * df['StdEpRet']) / reward_factor
    df['NormalizedAverageEpRet'] = df['AverageEpRet']  / reward_factor
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, sharey=False)
    g = g.map(plt.plot, "NormalizedAverageEpRet").add_legend()
    g = g.map(plt.fill_between, "Epoch", "LowEpRet", "HighEpRet" , **{'alpha':0.5}).add_legend()
    g.set_ylabels('AverageEpRet')

    df['NormalizedMaxEpRet'] = df['MaxEpRet'] / reward_factor
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, sharey=False)
    g = g.map(plt.plot, "Epoch", 'NormalizedMaxEpRet').add_legend()

    plt.figure()
    sns.lineplot(y='AverageVVals', x='Epoch', data=df, ci='sd')

    plt.figure()
    sns.lineplot(y='Entropy', x='Epoch', data=df, ci='sd')

    print(df.iloc[df['MaxEpRet'].to_numpy().argmax()]['best_design'])

    return df

def visualize_results(folder, x=None):
    
    if x is None:
        x = 'Epoch'
    
    df = load_exp_res(folder)
    df['seed'] = ['$%s$' %item for item in df['seed']]
    
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    ax = ax.ravel()
    sns.lineplot(x=x, y='MaxEpRet', data=df, hue='seed', ci='sd', legend=None, ax=ax[0])
    sns.lineplot(x=x, y='AverageEpRet', data=df, hue='seed', ci='sd', legend=None, ax=ax[1])
    sns.lineplot(x=x, y='Entropy', data=df, hue='seed', ci='sd',  ax=ax[2])
    sns.lineplot(x=x, y='EpLen', data=df, hue='seed', ci='sd',  ax=ax[3])

    best_designs = []
    for s in df['seed'].unique():
        best_designs.append(df[df['seed']==s]['best_design'].iloc[0])
        
    return best_designs

def extract_designs(best_designs):
    m0s = []
    x0s = []
    merits = []
    for i in range(len(best_designs)):
        design = best_designs[i][0].split(',')[0].split('|')
        m0 = [item.split(' ')[0] for item in design]
        x0 = [item.split(' ')[1] for item in design]
        merit = best_designs[i][0].split(',')[1].split(' ')[2]
        x0 = [int(item) for item in x0]

        m0s.append(m0)
        x0s.append(x0)
        merits.append(float(merit))
    
    return m0s, x0s, merits

def batch_finetune(df, env, max_thick=200):
    m0s, x0s, merits = extract_designs(np.unique(df['best_design'].to_numpy()).tolist())
    
    x_opts = []
    merits_opt = []
    for m0, x0 in tqdm(zip(m0s, x0s)):
        x_opt, res = finetune(env.simulator, m0, x0, env.target, bounds=[[15, max_thick]]*len(x0))
        merits_opt.append(1 - res.fun)
        x_opts.append(x_opt)
        print(merits, 1-res.fun)

    df = pd.DataFrame({'idx':list(range(len(merits))) * 2, 'group':['before finetune'] * len(merits) + ['after finetune'] * len(merits), 'Absorption':merits+merits_opt})

    sns.barplot(x='idx', y='Absorption', data=df, hue='group')
    # plt.ylim(0.9, 1.0)
    plt.axhline(np.max(merits_opt), linestyle='--', color='k')
    plt.title('Best absorption: {:.3f}'.format(np.max(merits_opt)))
    plt.show()

    sns.distplot(df[df['group']=='before finetune']['Absorption'], bins=5, kde=False)
    sns.distplot(df[df['group']=='after finetune']['Absorption'], bins=5, kde=False)
    plt.legend(['Before finetune', 'After finetune'])
    
    return x_opts, merits_opt

def select_subset(df, hparams, hvals):
    df_ = df.copy()
    for hparam, hval in zip(hparams, hvals):
        df_ = df_[df_[hparam] == hval]
    return df_

def compare_across_hparams(folder, hparams, abbrs):
    
    df = load_exp_res(folder)
    unique_hvals = []
    for h in hparams:
        unique_hvals.append(list(df[h].unique()))
        
    hparam_combs = list(product(*unique_hvals))
    legends = [' | '.join([abbr+':'+str(h) for abbr, h in zip(abbrs, item)]) for item in hparam_combs]
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    for i, hvals in enumerate(list(product(*unique_hvals))):
        df_ = select_subset(df, hparams, hvals)
        sns.lineplot(x='Epoch', y='AverageEpRet', ci='sd', hue=None, data=df_, ax=ax[0])
        sns.lineplot(x='Epoch', y='MaxEpRet', ci='sd', hue=None, data=df_, ax=ax[1])
        sns.lineplot(x='Epoch', y='Entropy', ci='sd', hue=None, data=df_, ax=ax[2])
        
        grouped_df = df_.groupby('Epoch')
        avg_mean, avg_std = grouped_df['AverageEpRet'].mean()[-10:].mean(), grouped_df['AverageEpRet'].std()[-10:].mean() # average of avgret over the last 10 epochs
        max_mean, max_std = grouped_df['MaxEpRet'].mean()[-10:].mean(), grouped_df['MaxEpRet'].std()[-10:].mean() # average of the maxret over the last 10 epochs
        best_mean, best_std = df_.groupby('seed')['MaxEpRet'].max().mean(), df_.groupby('seed')['MaxEpRet'].max().std()
        # print mean and std of average EpRet and MaxEpRet
        print('Exp {}, best ret {:.4f}+-{:.4f}, avg ret {:.4f}+-{:.4f}; max ret {:.4f}+-{:.4f}'.format(legends[i], best_mean, best_std, avg_mean, avg_std, max_mean, max_std))
    plt.legend(legends)
    plt.show()
    
    return df