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
    elif env_name =='CurrPerfectAbsorberVisNIR-v0':
        return perfect_absorber_vis_nir_curr(**kwargs)
    elif env_name == 'PianoBlack-v0':
        return piano_black(**kwargs)
    elif env_name == 'RadiativeCooler-v0':
        return radiative_cooler(**kwargs)
    elif env_name == 'IncandescentReflector-v0':
        return incandescent_reflector(**kwargs)
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
    # wavelengths = gen_grid(lamda_low, lamda_high)
    # wavelengths = np.concatenate((np.arange(0.38, 0.8, 0.005), np.arange(0.8, 2.02, 0.02)))
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    # materials = ['Cr', 'Ge', 'Si', 'TiO2', 'MgF2']
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
    # wavelengths = gen_grid(lamda_low, lamda_high)
    # wavelengths = np.concatenate((np.arange(0.38, 0.8, 0.005), np.arange(0.8, 2.02, 0.02)))
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)
    materials = ['Cr', 'Ge', 'Si', 'TiO2', 'MgF2']
    # materials = ['Ag', 'Al', 'Al2O3', 'Cr', 'Ge', 'HfO2', 'MgF2', 'Ni', 'Si', 'SiC', 'SiO2', 'Ti', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Fe2O3', 'Si3N4']
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

##################################### 
# Uniform black in [0.4-0.8] um #
#####################################


def piano_black(**kwargs):

    lamda_low = 0.4
    lamda_high = 0.9
    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.02)
    # materials = ['Al', 'Ge', 'Si', 'TiO2', 'MgF2']
    # Ti Ni ZnO Si TiO2 SiO2
    # materials = ['Ti', 'Ni', 'SiO2', 'TiO2', 'Si', 'ZnO']
    # materials = ['SiO2', 'TiO2', 'Si', 'ZnO', 'Ge', 'Cr', 'Ti', 'Ni', 'Al']
    # 'Fe2O3'
    materials = ['Ag', 'Al', 'Al2O3', 'Cr', 'Ge', 'HfO2', 'MgF2', 'Ni', 'Si', 'Si3N4', 'SiO2', 'Ti', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'PECVD-Si', 'Sputter-Si', 'Fe2O3']
    simulator = TMM_sim(materials, wavelengths, substrate='Air', substrate_thick=500)
    thickness_list = np.arange(15, 201, 5)

    # we maximize the total absorption in the whole wavelength range
    r = kwargs['R']
    target = {'R': np.ones_like(wavelengths) * r, 'A':np.ones_like(wavelengths) * (1-r)}

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              "spectrum_repr": kwargs['spectrum_repr'],
              "bottom_up": kwargs['bottom_up']}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make

#####################################
# Radiative cooler 
# Reflection = 1 in [0.3-2.5] um and [9.3, 10] um
# Reflection = 0 in [8-9.3] and [10, 13] um
#####################################

def radiative_cooler(**kwargs):

    thickness_list = np.arange(10, 2001, 100)

    wavelengths1 = np.arange(0.3, 2.5+eps, 0.05)
    wavelengths2 = np.arange(8, 9.3, 0.1)
    wavelengths3 = np.arange(9.3, 10, 0.02)
    wavelengths4 = np.arange(10, 13+0.1, 0.1)
    wavelengths = np.concatenate(
        (wavelengths1, wavelengths2, wavelengths3, wavelengths4))

    target1, target3 = np.ones_like(wavelengths1), np.ones_like(wavelengths3)
    target2, target4 = np.zeros_like(wavelengths2), np.zeros_like(wavelengths4)

    materials = ['HfO2', 'SiO2', 'Al2O3', 'MgF2', 'TiO2', 'SiC', 'Si3N4']
    simulator = TMM_sim(materials, wavelengths, substrate='Ag', substrate_thick=500)
    target = {'R':np.concatenate((target1, target2, target3, target4))}

    config = {'wavelengths': wavelengths,
                "materials": materials,
                'target': target,
                "merit_func": kwargs['merit_func'],
                "simulator": simulator,
                "spectrum_repr": kwargs['spectrum_repr'],
                "bottom_up": kwargs['bottom_up']}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make

#####################################
# Incandescent
#####################################

def incandescent_reflector(**kwargs):

    thickness_list = np.arange(10, 301, 10)

    wavelengths1 = np.arange(0.4, 0.7+eps, 0.005)
    wavelengths2 = np.arange(0.705, 3, 0.02)
    wavelengths = np.concatenate(
        (wavelengths1, wavelengths2))

    target1r, target2r = np.zeros_like(wavelengths1), np.ones_like(wavelengths2)
    target1t, target2t = np.ones_like(wavelengths1), np.zeros_like(wavelengths2)

    materials = ['HfO2', 'SiO2', 'Al2O3', 'MgF2', 'TiO2', 'SiC', 'Si3N4']
    simulator = TMM_sim(materials, wavelengths, substrate='Air', substrate_thick=500)
    target = {'R':np.concatenate((target1r, target2r)), 'T':np.concatenate((target1t, target2t))}

    config = {'wavelengths': wavelengths,
                "materials": materials,
                'target': target,
                "merit_func": kwargs['merit_func'],
                "simulator": simulator,
                "spectrum_repr": kwargs['spectrum_repr'],
                "bottom_up": kwargs['bottom_up']}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make



#####################################
# Curriculum Training, Perfect absorber in [0.4-2] um #
#####################################


def perfect_absorber_vis_nir_curr(**kwargs):

    lamda_low = 0.4
    lamda_high = 2.0
    thickness_list = np.arange(15, 201, 10)

    wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.05)
    materials = ['Cr', 'Ge', 'Si', 'TiO2', 'MgF2']
    simulator = TMM_sim(materials, wavelengths)

    # wavelengths = [np.arange(lamda_low, item+1e-3, 0.05) for item in lamda_high]
    # simulator = [TMM_sim(materials, item) for item in wavelengths]

    range1 = np.arange(0.4, 0.801, 0.05)
    range2 = np.arange(0.4, 1.401, 0.05)
    range3 = np.arange(0.4, 2.001, 0.05)
    ranges = [range1, range2, range3]

    # we maximize the total absorption in the whole wavelength range
    target = [{'A': np.ones_like(item)} for item in ranges]

    config = {'wavelengths': wavelengths,
              "materials": materials,
              'target': target,
              "merit_func": cal_reward,
              "simulator": simulator,
              "spectrum_repr": kwargs['spectrum_repr'],
              "switching_steps": [1e4, 3e4]}

    if kwargs['discrete_thick']:
        config['discrete_thick'] = True
        config['thickness_list'] = thickness_list

    def make():
        env = TMM(**config)

        return env

    return make


