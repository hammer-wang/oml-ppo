from spinup.utils.run_utils import ExperimentGrid
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.tasks import get_env_fn

def get_runner(args):
    
    if args.exp == 'pa_compare':
        return pa_compare_models(args)

def pa_compare_models(args):
    '''
    Compare three model architectures on the perfect absorber task
    '''

    env_kwargs = {"discrete_thick":True, 'spectrum_repr':False, 'bottom_up':False}
    env = 'PerfectAbsorberVisNIR-v0'

    eg = ExperimentGrid(name=args.name)
    eg.add('env_fn', get_env_fn(env, **env_kwargs))
    eg.add('seed', [42*(i+1) for i in range(args.num_runs)])
    eg.add('epochs', 2000)
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(64,)], 'hid')
    eg.add('ac_kwargs:cell_size', 64, '')
    eg.add('ac_kwargs:not_repeat', [False, True])
    eg.add('ac_kwargs:ortho_init', ['on'])
    eg.add('ac_kwargs:hierarchical', [False, True])
    eg.add('ac_kwargs:channels', 16)
    eg.add('use_rnn', True)
    eg.add('gamma', 1)
    eg.add('beta', [0.01])
    eg.add('lam', [0.95])
    eg.add('max_ep_len', 10, in_name=True)
    eg.add('actor_critic', core.RNNActorCritic)
    eg.add("train_pi_iters", [5])
    eg.add("pi_lr", [5e-5])
    eg.add('reward_factor', [1])
    eg.add('spectrum_repr', [False])
    eg.add('ac_kwargs:scalar_thick', [False])

    return eg