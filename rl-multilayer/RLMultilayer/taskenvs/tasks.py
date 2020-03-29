"""
Return tasks
"""
import numpy as np
from RLMultilayer.taskenvs.task_envs import TMM
from RLMultilayer.utils import TMM_sim, cal_reward
import gym

eps=1e-5


def get_env_fn(env_name, **kwargs):

    if env_name == 'PerfectAbsorberVis-v0':
        return perfect_absorber_vis(**kwargs)
    elif env_name == 'PerfectAbsorberVisNIR-v0':
        return perfect_absorber_vis_nir(**kwargs)
    elif env_name == 'PerfectAbsorberVisNIR-v1':
        return perfect_absorber_vis_nir5(**kwargs)
    else:
        try:
            return lambda: gym.make(env_name)
        except:
            raise NotImplementedError("Env not registered!")

#####################################
# Perfect absorber in visible range #
#####################################


def perfect_absorber_vis(**kwargs):

    lamda_low = 0.4
    lamda_high = 0.8
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.02)
    materials = ['Ag', 'Al', 'Cr', 'Ge', 'SiC',
                 'HfO2', 'SiO2', 'Al2O3', 'MgF2', 'TiO2']
    simulator = TMM_sim(materials, wavelengths)
    thickness_list = np.arange(15, 201, 2)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make

#####################################
# Perfect absorber in [0.4-2] um #
#####################################

def perfect_absorber_vis_nir(**kwargs):

    def gen_grid(low, high):
    
        ws = [low]
        while ws[-1] < high:
            ws.append(1.03*ws[-1])

        return np.array(ws)

    lamda_low = 0.4
    lamda_high = 2.0
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    materials = ['Ag', 'Al', 'Al2O3', 'Cr', 'Ge', 'HfO2', 'MgF2', 'Ni', 'Si', 'SiO2', 'Ti', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Fe2O3']
    simulator = TMM_sim(materials, wavelengths, substrate='Glass', substrate_thick=500)
    thickness_list = np.arange(15, 201, 5)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              **kwargs}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make


def perfect_absorber_vis_nir5(**kwargs):

    def gen_grid(low, high):
    
        ws = [low]
        while ws[-1] < high:
            ws.append(1.03*ws[-1])

        return np.array(ws)

    lamda_low = 0.4
    lamda_high = 2.0
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    materials = ['Cr', 'Ge', 'Si', 'TiO2', 'MgF2']
    simulator = TMM_sim(materials, wavelengths, substrate='Glass', substrate_thick=500)
    thickness_list = np.arange(15, 201, 5)

    # we maximize the total absorption in the whole wavelength range
    target = {'A': np.ones_like(wavelengths)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              **kwargs}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make

