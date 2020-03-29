import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from copy import deepcopy


class TMM(gym.Env):
    '''
    TMM module for calculating the reward. 
    '''

    def __init__(self, **kwargs):
        self.layers = []
        self.thicknesses = []
        self.current_merit = 0
        self.kwargs = kwargs
        self.simulator = kwargs['simulator']
        self.merit_func = kwargs['merit_func']
        self.target = kwargs['target']
        self.spectrum_repr=  kwargs['spectrum_repr'] # whether to add spectrum to the representation
        self.use_curr = False
        self.scalar_thick = None
    
        if isinstance(kwargs['target'], list):
            self.curr_idx = 0
            self.use_curr = True # use curriculum 
            # self.simulator = kwargs['simulator'][self.curr_idx]
            self.target = kwargs['target'][self.curr_idx]
            self.switching_steps = kwargs['switching_steps']

        # self.materials = list(self.simulator.nk_dict.keys())
        self.materials = list(self.simulator.mats)
        self.materials_idx = dict(zip(self.materials, range(len(self.materials))))
        self.num_materials = len(self.materials)
        self.step_idx = 0
        self.material_dim = self.num_materials + 1

        self.discrete_thickness = kwargs['discrete_thick']
        
        if self.discrete_thickness:
            self.thickness_list = kwargs['thickness_list']
            self.num_thicknesses = len(self.thickness_list)

            observation_list = [spaces.Discrete(self.material_dim),
            spaces.Discrete(self.num_thicknesses)]
            action_list = [spaces.Discrete(self.material_dim),
            spaces.Discrete(self.num_thicknesses)]
            
            # add spectrum representation to the observation list
            if self.spectrum_repr:
                self.spectrum_dim = (3, len(self.simulator.wavelength))
                self.spectrum_dim_flat = np.prod(self.spectrum_dim)
                observation_list.append(spaces.Box(low=0., high=1.,shape=self.spectrum_dim))

            self.observation_space = spaces.Tuple(observation_list)
        
        else:
            # TODO: update the spectrum_repr part
            self.num_thicknesses = 1
            self.observation_space = spaces.Tuple([
                spaces.Discrete(self.num_materials+1),
                spaces.Box(low=0, high=1, shape=(1,))
            ])
        
        # self.obs_dim = self.num_materials + self.num_thicknesses + 1
        
        self.obs_dim = self.num_thicknesses + self.material_dim
        self.act_dim = self.obs_dim
        if self.spectrum_repr:
            self.obs_dim = self.obs_dim + self.spectrum_dim_flat

        self.action_space = spaces.Tuple(action_list)
        self.max_wavelength = self.simulator.wavelength[-1] * 1e3 # nm

    def step(self, action):
        
        self.step_idx += 1

        if self.discrete_thickness:
            mat_idx, thick_idx = int(action[0]), int(action[1])
            thickness = self.thickness_list[thick_idx]
            obs = np.zeros((self.obs_dim), dtype=np.float32)
            obs[thick_idx+self.material_dim] = 1

        else:
            mat_idx, thickness = int(action[0]), action[1]
            thickness = int(min(max(15, (thickness+0.5) * 200), 200))

            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[mat_idx] = 1
            obs[self.num_materials+1] = action[1]
        

        # EOS = self.material_dim - 2 if self.not_repeat else self.material_dim - 1
        EOS = self.material_dim - 1

        if mat_idx == EOS and len(self.layers) >= 1:
            episode_over = True

            # for curriculum training
            if self.use_curr and self.curr_idx < len(self.switching_steps):
                if self.step_idx >= self.switching_steps[self.curr_idx]:
                    print('switch to task {}'.format(self.curr_idx+1))
                    self.curr_idx += 1
                    # self.simulator = self.kwargs['simulator'][self.curr_idx]
                    self.target = self.kwargs['target'][self.curr_idx]

            R, T, A = self.simulator.spectrum(self.layers, [np.inf] + self.thicknesses + [np.inf])
            reward = self.merit_func(R, T, A, self.target)

            return obs, reward, True, {}
        else:
            # 
            # mat_idx = mat_idx if len(self.layers) >= 1 and self.materials_idx[self.layers[-1]] > mat_idx else mat_idx + 1

            # avoid generating the same structures
            # if self.not_repeat:
            #     if len(self.layers) == 0:
            #         pass
            #     elif self.materials_idx[self.layers[-1]] <= mat_idx:
            #         mat_idx += 1
            #     else:
            #         pass
            obs[mat_idx] = 1

            # if mat_idx >= len(self.materials):
            #     import pdb; pdb.set_trace()
            material = self.materials[mat_idx]
            episode_over = False

        if not self.kwargs['bottom_up']:
            self.layers.append(material)
        else:
            self.layers.insert(0, material)
            
        self.thicknesses.append(thickness)
        # R, T, A = self.simulator.spectrum(self.layers, [np.inf] + self.thicknesses + [np.inf])

        if self.spectrum_repr:
            obs[self.act_dim:] = np.concatenate((R,T,A))
        
        # TODO: add a automatically switching merit function
        # merit = self.merit_func(R, T, A, self.target)
        # reward = merit - self.current_merit
        reward = merit =  0

        # penalize outputtting the same layers consecutively
        # if len(self.layers) > 1 and self.layers[-1] == self.layers[-2]:
        #     reward -= 0.1

        self.current_merit = merit

        if self.scalar_thick:
            obs = self.get_scalar_repr(obs)

        return obs, reward, episode_over, {}

    def reset(self):
        self.layers = []
        self.thicknesses = []
        self.current_merit = 0

        init_obs = np.zeros(self.obs_dim, dtype=np.float32)
        # if self.spectrum_repr:
        #     init_spectrum = np.zeros(self.spectrum_dim)
        #     init_spectrum[1] = 1
        #     init_obs = np.concatenate((init_obs, init_spectrum.reshape(-1)))
        # import pdb; pdb.set_trace()
        if self.scalar_thick:
            init_obs = self.get_scalar_repr(init_obs)
            init_obs[-1] = 0

        return init_obs

    def render(self):

        self.simulator.spectrum(
            self.layers, [np.inf] + self.thicknesses + [np.inf],
            plot=True, title=True)

    def get_scalar_repr(self, obs):
        # import pdb; pdb.set_trace()
        thick= self.thickness_list[obs[self.num_materials+1:self.num_materials+1+self.num_thicknesses].argmax()] / self.thickness_list[-1]
        obs = np.concatenate((obs[:self.num_materials+1], np.array([thick])))

        return obs

    def update_with_ac(self, **kwargs):
        
        if 'scalar_thick' in kwargs:
            self.scalar_thick = False
        if 'not_repeat' in kwargs:
            self.not_repeat = kwargs['not_repeat']
