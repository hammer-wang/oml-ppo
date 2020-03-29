import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete, Tuple
import copy

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def tuple_space_dim(obs, action=False):
    '''
    Extract observaiton/action dimensions from space.Tuple object.

    Currently this function only deals with 1d inputs, i.e., all elements in 
    the Tuple has 1 dimension 
    '''
    obs_dim = []
    for item in obs:
        if isinstance(item, Discrete):
            # for obs, we need to store one-hot representation
            dim = 1 if action else item.n
            obs_dim.append(dim)
        else:
            obs_dim.append(np.prod(item.shape))

    return sum(obs_dim)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def build_actor(action_space, hidden_sizes, activation, obs_dim, **kwargs):
    pis = []

    def build_one_head(item, i):
        if isinstance(item, Box):
            pi = MLPGaussianActor(
                obs_dim, item.shape[0], hidden_sizes, activation)
        elif isinstance(item, Discrete):
            if i > 0 and kwargs['hierarchical']:
                if not kwargs['act_emb']:
                    pi = MLPCategoricalActor(
                    obs_dim+action_space[0].n, item.n, hidden_sizes, activation)
                else: 
                    pi = MLPCategoricalActor(
                    obs_dim+kwargs['act_emb_dim'], item.n, hidden_sizes, activation)
            else:
                pi = MLPCategoricalActor(
                    obs_dim, item.n, hidden_sizes, activation)
        return pi

    if isinstance(action_space, Tuple):
        for i, item in enumerate(action_space):
            pis.append(build_one_head(item, i))
    else:
        pis.append(build_one_head(action_space))

    return nn.ModuleList(pis)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, zero_idx=None):

        # zero_idx: which index to zero out
        logits = self.logits_net(obs)
        if zero_idx is not None:
            mask = torch.eye(len(logits))
            idx = list(range(len(mask)))
            idx.remove(zero_idx.item())
            logits = torch.matmul(logits, mask[:, idx])

        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -3 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +
                          [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh, **kwargs):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        if isinstance(observation_space, Tuple):
            # concatenate observation space
            obs_dim = tuple_space_dim(observation_space)
        else:
            obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        self.pi = build_actor(action_space, hidden_sizes, activation, obs_dim)
        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, *args):
        
        num_heads = len(self.pi)

        act_ls = []
        with torch.no_grad():
            logp_a = 0
            for i in range(num_heads):
                dist = self.pi[i]._distribution(obs)
                act = dist.sample()
                logp_a += self.pi[i]._log_prob_from_distribution(dist, act)
                act_ls.append(act.numpy().ravel())
        
            act = np.concatenate(act_ls)
            v = self.v(obs)
        # import pdb; pdb.set_trace()
        return act.squeeze(), v.numpy(), logp_a.numpy(), np.array([1])

    def act(self, obs, *args):
        return self.step(obs)[0]
    
class RNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                hidden_sizes=(64, 64), activation=nn.Tanh, **kwargs):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        if isinstance(observation_space, Tuple):

            if not kwargs['scalar_thick']:
                # concatenate observation space
                self.obs_dim = obs_dim = tuple_space_dim(observation_space)
            else:
                self.obs_dim = obs_dim = self.observation_space[0].n + 1

        else:
            self.obs_dim = obs_dim = observation_space.shape[0]

        if isinstance(action_space, Tuple):
            # concatenate observation space
            self.act_dim = act_dim = tuple_space_dim(action_space)
        else:
            self.act_dim = act_dim = action_space.shape[0]

        # whether to use cnn to process input
        self.use_cnn = True if obs_dim > act_dim else False

        cell_size = kwargs['cell_size']

        # 1d cnn filter size
        if self.use_cnn:
            channels = kwargs['channels']
            self.cnn = nn.Sequential(nn.Conv1d(3, channels//2, 3),
                                     nn.ReLU(),
                                     nn.MaxPool1d(channels//2, stride=2),
                                     nn.Conv1d(channels//2, channels, 3),
                                     nn.AdaptiveMaxPool1d(3),
                                     nn.ReLU(),
                                     nn.Flatten(),
                                     nn.Linear(channels*3, 16)
                                     )
        
        if self.use_cnn:
            self.rnn = nn.GRUCell(16 + act_dim, cell_size)
        else:
            self.rnn = nn.GRUCell(obs_dim, cell_size)

        self.not_repeat = kwargs['not_repeat']
        self.hierarchical = kwargs['hierarchical']
        self.act_emb = kwargs['act_emb']
        self.act_emb_dim = kwargs['act_emb_dim']

        # policy builder depends on action space
        self.pi = build_actor(action_space, hidden_sizes, activation, cell_size, **kwargs)

        # build value function
        self.v = MLPCritic(cell_size, hidden_sizes, activation)

        if self.act_emb and self.hierarchical:
            self.emb = nn.Embedding(self.action_space[0].n, self.act_emb_dim)
        # import pdb; pdb.set_trace()

        if kwargs['ortho_init'] is 'on':
            self._orthogonal_init()
            
    def step(self, obs, hidden, evaluate=False):
        '''
        Args:
            obs: torch.tensor, current observation.
            hidden: torch.tensor. Hidden state from previous time. 
        '''

        # extract material of the previous layer (lower level action)
        if torch.sum(obs) > 0: # not the beginning of generation
            prev_act = torch.argmax(obs[:self.observation_space[0].n])
        else:
            prev_act = None
        
        # number of actions
        num_heads = len(self.pi)
        
        act_ls = []
        with torch.no_grad():
            logp_a = 0
            hidden = self._get_hidden(obs, hidden)

            shift = False
            for i in range(num_heads):
                
                zero_idx = self._mask_idx(i, obs, prev_act)

                if i == 0:
                    dist = self.pi[i]._distribution(hidden, zero_idx=zero_idx)
                    try:
                        act = dist.sample()
                    except:
                        import pdb; pdb.set_trace()
                else:
                    if self.hierarchical:
                        lower_act = torch.zeros(self.action_space[0].n)
                        if self.act_emb:
                            if shift:
                                # lower_act[act] = dist.log_prob(act-1)
                                lower_act = self.emb(act.to(torch.long))
                            else:
                                lower_act = self.emb(act.to(torch.long))
                                # lower_act[act] = dist.log_prob(act)
                        else:
                            lower_act[act] = 1
                        dist = self.pi[i]._distribution(torch.cat((hidden, lower_act)), zero_idx=None)
                    else:
                        dist = self.pi[i]._distribution(hidden, zero_idx=None)
                    act = dist.sample()

                logp_a += self.pi[i]._log_prob_from_distribution(dist, act)

                if i==0 and self.not_repeat and zero_idx <= act:
                    act += 1
                    shift = True

                act_ls.append(act.numpy().ravel())
        
            act = np.concatenate(act_ls)
            v = self.v(hidden)

        return act.squeeze(), v.numpy(), logp_a.numpy(), hidden.numpy()

    def act(self, obs, hidden):
        return self.step(obs, hidden)[0]

    def evaluate(self, obs, act, hidden):

        if torch.sum(obs) > 0: # not the beginning of generation
            prev_act = torch.argmax(obs[:self.observation_space[0].n])
        else:
            prev_act = None

        num_heads = len(self.pi)
        pi_ls = []
        logp_a = 0

        hidden = self._get_hidden(obs, hidden)

        shift = False
        for i in range(num_heads):

            zero_idx = self._mask_idx(i, obs, prev_act)

            if i == 0: # for material selection, no need to concatenate
                dist = self.pi[i]._distribution(hidden, zero_idx=zero_idx)
            else:
                # there may not be gradient
                if self.hierarchical:
                    lower_act = torch.zeros(self.action_space[0].n)
                    act_ = act[0].to(torch.long)
                    if self.act_emb:
                        if shift:
                            # lower_act[act_] = dist.log_prob(act_-1)
                            lower_act = self.emb(act_.to(torch.long))
                        else:
                            # lower_act[act_] = dist.log_prob(act_)
                            lower_act = self.emb(act_.to(torch.long))
                    else:
                        lower_act[act_] = 1
                    dist = self.pi[i]._distribution(torch.cat((hidden, lower_act)), zero_idx=None)
                else: 
                    dist = self.pi[i]._distribution(hidden, zero_idx=zero_idx)

            if len(act.size()) > 0:
                if i==0 and self.not_repeat and zero_idx < act[0]:
                    # compensate for the index shifting.
                    logp_a += self.pi[i]._log_prob_from_distribution(dist, act[i]-1)
                    shift = True
                else:
                    logp_a += self.pi[i]._log_prob_from_distribution(dist, act[i])
            else: 
                logp_a += self.pi[i]._log_prob_from_distribution(dist, act)
            pi_ls.append(dist)
    
        v = self.v(hidden)
        ent = torch.stack([[pi.entropy().squeeze() for pi in pi_ls][0]]).sum()

        return v.view(1), logp_a.view(1), hidden, ent

    def _orthogonal_init(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)

    def _get_hidden(self, obs, hidden):

        if self.use_cnn:
            spectrum = obs[self.act_dim:].reshape(1, 3, -1)
            spe_hid = self.cnn(spectrum)
            inp = torch.cat((obs[:self.act_dim].reshape(1, -1), spe_hid), dim=1)
            hidden = self.rnn(inp, hidden.reshape(1, -1)).squeeze()

        else:
            hidden = self.rnn(obs.reshape(1, -1), hidden.reshape(1, -1)).squeeze()

        return hidden

    def _mask_idx(self, i, obs, prev_act):

        # find the materials of the previous layer
        if i == 0:
            if self.not_repeat and torch.sum(obs) > 0:
                zero_idx = prev_act
            elif torch.sum(obs) == 0:
                # not terminate at the beginning
                zero_idx = torch.tensor(self.observation_space[0].n-1)
            else:
                zero_idx = None
        else:
            zero_idx = None

        return zero_idx


        
