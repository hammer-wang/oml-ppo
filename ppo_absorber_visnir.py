from spinup.utils.run_utils import ExperimentGrid
from RLMultilayer.algos.ppo.ppo import ppo
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.tasks import get_env_fn
from RLMultilayer.utils import cal_reward
import torch

import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--env', type=str, default='PerfectAbsorberVisNIR-v0')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--discrete_thick', action="store_true")
    parser.add_argument('--maxlen', default=5, type=int)
    parser.add_argument('--hierarchical', action='store_true', help='if set to true, then output out the material type first, then condition the material thickness on the material type')
    parser.add_argument('--use_rnn', action='store_true')
    parser.add_argument('--spectrum_repr', action='store_true')
    args = parser.parse_args()

    env_kwargs = {"discrete_thick":args.discrete_thick, 'spectrum_repr':args.spectrum_repr, 'bottom_up':False, 'merit_func':cal_reward}

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', get_env_fn(args.env, **env_kwargs))
    eg.add('seed', [42*(i+1) for i in range(args.num_runs)])
    eg.add('epochs', 3000)
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(64,)], 'hid')
    eg.add('ac_kwargs:cell_size', 64, '')
    eg.add('ac_kwargs:not_repeat', [True, False])
    eg.add('ac_kwargs:ortho_init', ['on'])
    eg.add('ac_kwargs:hierarchical', [True, False])
    eg.add('ac_kwargs:channels', 16)
    eg.add('ac_kwargs:act_emb', [True])
    eg.add('ac_kwargs:act_emb_dim', 5)
    eg.add('use_rnn', args.use_rnn)
    eg.add('gamma', 1)
    eg.add('beta', [0.01])
    eg.add('lam', [0.95])
    eg.add('max_ep_len', [args.maxlen], in_name=True)
    eg.add('actor_critic', core.RNNActorCritic if args.use_rnn else core.MLPActorCritic)
    eg.add("train_pi_iters", [5])
    eg.add("pi_lr", [5e-5])
    eg.add('reward_factor', [1])
    eg.add('spectrum_repr', [args.spectrum_repr])
    eg.add('ac_kwargs:scalar_thick', [False])



    eg.run(ppo, num_cpu=args.cpu, data_dir='./Experiments/{}'.format(args.exp_name), datestamp=False)


