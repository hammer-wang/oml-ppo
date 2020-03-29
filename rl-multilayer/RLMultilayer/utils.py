from mpi4py import MPI
import matplotlib
from tmm import coh_tmm
import pandas as pd
import os
from numpy import pi
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from scipy.optimize import minimize
import json
from tqdm import tqdm

DATABASE = './data'
INSULATORS = ['HfO2', 'SiO2', 'SiC', 'Al2O3', 'MgF2', 'TiO2', 'Fe2O3', 'MgF2', 'Si3N4', 'TiN', 'ZnO', 'ZnS', 'ZnSe']
METALS = ['Ag', 'Al', 'Cr', 'Ge', 'Si', 'Ni']

num_workers = 8

def cal_reward(R, T, A, target):
    '''
    Calculate reward based on given spectrums. 
    We calculate the reward using averaged (1-mse).

    Args:
        R, T, A: numpy array. Reflection, transmission, and 
        absorption spectrums, respectively.
        target: dict. {'R':np.array, 'T':np.array, 'A':np.array}

    Returns:
        reward: float. Reward for the spectrum. 
    '''

    reward = 0
    for k, v in target.items():

        if k == 'R':
            res = R
        elif k == 'T':
            res = T
        else:
            res = A
        
        reward += 1 - np.abs(res.squeeze() - v).mean()

    reward /= len(target)

    return reward


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def batch_spectrum(env, names_list, thickness_list):

    def spectrum(args):
        '''
        Inputs: 
            1. names: list of lists, each list correspond to the structures
            2. thickness: list of lists
        '''
        names, thickness = args
        R, T, A = env.spectrum(names, thickness, 0, False)

        return R, T, A

    res = Parallel(n_jobs=num_workers)(delayed(spectrum)(args)
                                       for args in
                                       zip(names_list, thickness_list))
    res = np.array(res)
    Rs, Ts, As = res[:, 0, :], res[:, 1, :], res[:, 2, :]

    return Rs, Ts, As


def merge_layers(categories, thicknesses):
    '''
    Merges consecutive layers with the same material types.
    '''

    thicknesses = thicknesses[1:-1]
    c_output = [categories[0]]
    t_output = [thicknesses[0]]
    for i, (c, d) in enumerate(zip(categories[1:], thicknesses[1:])):

        if c == c_output[-1]:
            t_output[-1] += d
            continue
        else:
            c_output.append(c)
            t_output.append(d)

    t_output.insert(0, np.inf)
    t_output.insert(len(t_output), np.inf)

    return c_output, t_output


def get_structure(categories, values, materials, ds, continuous=False,
                  max_value=400):
    '''
    Given categories and values, return the strucure in the form 
    (name (str), thickness (nm))
    '''

    def threshold(value):
        '''

        '''

    names = [materials[item] for item in categories]

    if not continuous:
        thickness = [np.inf] + [ds[item] for item in values] + [np.inf]
    else:
        thickness = []
        for category, value in zip(categories, values):
            name = materials[category]
            if name == 'Ag':
                thickness.append(
                    min(max(15, int(value * max_value//2)), max_value))
            elif name in METALS:
                thickness.append(
                    min(max(5, int(value * max_value//2)), max_value))
            elif name in INSULATORS:
                thickness.append(
                    min(max(1, int(value * max_value//2)), max_value))
            else:
                raise ValueError('Material not known')
        # thickness = [np.inf] + [min(max(5, int(item * 2e2)), 200) for i,
        # item in enumerate(values)] + [np.inf]
        thickness = [np.inf] + thickness + [np.inf]
    return names, thickness

class DesignTracker():
    def __init__(self, epochs, **kwargs):
        """
        This class tracks the best designs discovered.
        """
        if epochs == -1:
            self.layer_ls = []
            self.thick_ls = []
            self.max_ret_ls = []
        self.layer_ls = [0] * epochs
        self.thick_ls = [0] * epochs
        self.max_ret_ls = [0] * epochs
        self.kwargs = kwargs
        self.current_e = 0

    def store(self, layers, thicknesses, ret, e, append_mode=False):
        
        if append_mode:
            self.layer_ls.append(layers)
            self.thick_ls.append(thicknesses)
            self.max_ret_ls.append(ret)

        else:
            if ret >= self.max_ret_ls[e]:
                self.layer_ls[e] = layers
                self.thick_ls[e] = thicknesses
                self.max_ret_ls[e] = ret

    def save_state(self):
        # save buffer from all processes
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        filename = os.path.join(self.kwargs['output_dir'], 'design_tracker_{}.pkl'.format(rank))
        pkl.dump(self, open(filename, 'wb'))
    
    def print_progress(self):
        progress = list(zip(self.layer_ls,  self.thick_ls, self.max_ret_ls))
        read_progress = []
        for i in range(len(progress)):
            if progress[i] == (0,0,0):
                break
            read_progress.append(['|'.join([l + ' ' + str(d) + ' nm' for l, d in zip(progress[i][0], progress[i][1])]) + ', Merit {:.3f}'.format(progress[i][2])])

        return read_progress

def print_progress(progress):

    for i in range(len(progress)):
        print(progress[i], 0)
        progress[i] = ['|'.join([l + ' ' + str(d) + ' nm' for l, d in zip(progress[i][0], progress[i][1])]), progress[i][2]]

    return progress

class TMM_sim():
    def __init__(self, mats=['Ge'], wavelength=np.arange(0.38, 0.805, 0.01), substrate='Cr', substrate_thick=500):
        '''
        This class returns the spectrum given the designed structures.
        '''
        self.mats = mats
        # include substrate
        self.all_mats = mats + [substrate] if substrate not in ['Glass', 'Air'] else mats
        self.wavelength = wavelength
        self.nk_dict = self.load_materials()
        self.substrate = substrate
        self.substrate_thick = substrate_thick

    def load_materials(self):
        '''
        Load material nk and return corresponding interpolators.

        Return:
            nk_dict: dict, key -- material name, value: n, k in the 
            self.wavelength range
        '''
        nk_dict = {}

        for mat in self.all_mats:
            nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
            nk.dropna(inplace=True)
            wl = nk['wl'].to_numpy()
            index = (nk['n'] + nk['k'] * 1.j).to_numpy()
            mat_nk_data = np.hstack((wl[:, np.newaxis], index[:, np.newaxis]))


            mat_nk_fn = interp1d(
                    mat_nk_data[:, 0].real, mat_nk_data[:, 1], kind='quadratic')
            nk_dict[mat] = mat_nk_fn(self.wavelength)

        return nk_dict

    def spectrum(self, materials, thickness, theta=0, plot=False, title=False):
        '''
        Input:
            materials: list
            thickness: list
            theta: degree, the incidence angle

        Return:
            s: array, spectrum
        '''
        degree = pi/180
        if self.substrate != 'Air':
            thickness.insert(-1, self.substrate_thick) # substrate thickness

        R, T, A = [], [], []
        for i, lambda_vac in enumerate(self.wavelength * 1e3):

            # we assume the last layer is glass
            if self.substrate == 'Glass':
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1.45, 1]
            elif self.substrate == 'Air':
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1]
            else:
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict[self.substrate][i], 1]

            # n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict['Cr'][i]]

            # mport pdb; pdb.set_trace()
            res = coh_tmm('s', n_list, thickness, theta * degree, lambda_vac)
            R.append(res['R'])
            T.append(res['T'])

        R, T = np.array(R), np.array(T)
        A = 1 - R - T

        if plot:
            self.plot_spectrum(R, T, A)
            if title:
                thick = thickness[1:-1]
                title = ' | '.join(['{}nm {}'.format(d, m)
                                    for d, m in zip(thick, materials)])
                if self.substrate is not 'Air':
                    title = 'Air | ' + title + ' | {}nm {} '.format(self.substrate_thick, self.substrate) + '| Air'
                else:
                    title = 'Air | ' + title + ' | Air'
                plt.title(title, **{'size': '10'})

        return R, T, A

    def plot_spectrum(self, R, T, A):

        plt.plot(self.wavelength * 1000, R, self.wavelength *
                 1000, T, self.wavelength * 1000, A, linewidth=3)
        plt.ylabel('R/T/A')
        plt.xlabel('Wavelength (nm)')
        plt.legend(['R: Average = {:.2f}%'.
                    format(np.mean(R)*100),
                    'T: Average = {:.2f}%'.
                    format(np.mean(T)*100),
                    'A: Average = {:.2f}%'.
                    format(np.mean(A)*100)])
        plt.grid('on', linestyle='--')
        plt.ylim([0, 1])


# Plotting utils
def visualize_progress(file, x, ax=None, color='b', alpha=1):
    df = pd.read_csv(file, sep="\t")
    width = 0.5
    # x = 'Time'
    if ax is None:
        fig, ax = plt.subplots(2,1)
    sns.lineplot(x=x, y='MaxEpRet', data=df, ax=ax[0], color=color, alpha=alpha)
    # ax[0].legend(['Max {}'.format(np.max(df['MaxEpRet']))])
    sns.lineplot(x=x, y='AverageEpRet', data=df,
                 ax=ax[1], color=color, alpha=alpha)
    plt.fill_between(df[x],
                     df['AverageEpRet']-width/2*df['StdEpRet'],
                     df['AverageEpRet']+width/2*df['StdEpRet'],
                     alpha=0.3, color=color)

    return df

def combine_tracker(folder):
    '''
    Merge all buffers
    '''
    trackers = []
    
    if 'design_tracker_merged.pkl' in os.listdir(folder):
        tracker_file = os.path.join(folder, 'design_tracker_merged.pkl')
        combined_tracker = pkl.load(open(tracker_file, 'rb'))
        return combined_tracker

    for file in os.listdir(folder):
        if file.startswith('design_tracker_'):
            tracker_file = os.path.join(folder, file)
            trackers.append(pkl.load(open(tracker_file, 'rb')))        

    combined_tracker = DesignTracker(len(trackers[0].layer_ls))
    max_idx = np.argmax(np.array([tracker.max_ret_ls for tracker in trackers]), axis=0)
    for e in range(len(trackers[0].layer_ls)):
        combined_tracker.layer_ls[e] = trackers[max_idx[e]].layer_ls[e]
        combined_tracker.thick_ls[e] = trackers[max_idx[e]].thick_ls[e]
        combined_tracker.max_ret_ls[e] = trackers[max_idx[e]].max_ret_ls[e]
    
    if combined_tracker.layer_ls[-1] != 0:
        tracker_file = os.path.join(folder, 'design_tracker_merged.pkl')
        pkl.dump(combined_tracker, open(os.path.join(folder, tracker_file), 'wb'))

    return combined_tracker

def summarize_res(exp_ls, seed_ls, color, alpha, x='Epoch'):
        
    root = '../spinningup/data/'
    progress_ls = []
    max_ret_ls = []

    params = {'size':14}
    matplotlib.rc('font', **params)

    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for a, c, exp, seed in zip(alpha, color, exp_ls, seed_ls):
        folder = os.path.join(root, exp, exp+'_s{}'.format(seed))
        progress_file = os.path.join(folder, 'progress.txt')
        df = visualize_progress(progress_file, x=x, ax=ax, color=c, alpha=a)

        tracker = combine_tracker(folder)
        progress = tracker.print_progress()
        print('{}, Best discovered so far {}'.format(exp, progress[np.argmax(tracker.max_ret_ls)]))
        progress_ls.append(progress)
        max_ret_ls.append('Max merit {:.3f}'.format(np.max(df['MaxEpRet'])))

    ax[0].legend(max_ret_ls)
    ax[1].legend(exp_ls)
    plt.show()
    return progress_ls

def load_exp_res(folder):
    subfolders = [item for item in glob.glob(folder+'/*')]

    def read_hyper(file_name, rep=10):

        with open(os.path.join(file_name, 'config.json')) as f:
            hypers = json.load(f)
            hypers_dict = {}
            for k, v in hypers.items():
                if k.startswith('logger'):
                    continue
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, list):
                            hypers_dict[str(k)+'_'+str(kk)] = [vv[0]]*rep
                        else:
                            hypers_dict[str(k)+'_'+str(kk)] = [vv]*rep
                else: 
                    hypers_dict[k] = [v] * rep
            
            hyper_df = pd.DataFrame(hypers_dict)
            return hyper_df 

    first=True # first pandas file to load
    for subfolder in tqdm(subfolders):
        runs = glob.glob(subfolder+'/*')
        num_epochs = len(pd.read_csv(os.path.join(runs[0], 'progress.txt'),sep='\t'))
        for run in runs:

            tracker = combine_tracker(run)
            progress = tracker.print_progress()
            best_design = progress[np.argmax(tracker.max_ret_ls)]

            if first:
                df = pd.read_csv(os.path.join(run, 'progress.txt'),sep='\t')
                hyper_df = read_hyper(run, rep=len(df))
                best_designs_df = pd.DataFrame([{'best_design':best_design}]*len(df))
                df = pd.concat([df, hyper_df, best_designs_df], axis=1)
                first = False

            else:
                df_ = pd.read_csv(os.path.join(run, 'progress.txt'),sep='\t')
                hyper_df = read_hyper(run, rep=len(df_))
                best_designs_df = pd.DataFrame([{'best_design':best_design}]*len(df_))
                df_ = pd.concat([df_, hyper_df, best_designs_df], axis=1)
                df = pd.concat([df, df_], axis=0)   

    return df   


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
        bounds = [(15, 200)] * len(x0)
    
    res = minimize(objective_func, x0, bounds=bounds, options={'disp':True})
    x_opt = [int(item) for item in res.x]
    
    if display:
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x0+[np.inf], title=True, plot=True)
        plt.figure()
        simulator.spectrum(m0, [np.inf]+x_opt+[np.inf], title=True, plot=True)
    
    return x_opt, res
