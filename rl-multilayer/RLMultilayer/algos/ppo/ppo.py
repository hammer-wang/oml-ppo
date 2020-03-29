# Add RLMultilayer package
import sys
# sys.path.insert(0, "/Users/wanghaozhu/Documents/Projects/RL_multilayer")
from RLMultilayer.taskenvs.tasks import get_env_fn
from RLMultilayer.algos.ppo import core
from RLMultilayer.taskenvs.task_envs import TMM
from RLMultilayer.utils import DesignTracker
from mpi4py import MPI

from torchsummary import summary

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# other dependencies
import numpy as np
from torch.optim import Adam
import torch
from gym import spaces
import gym
import time
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, cell_size=None):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        if cell_size:
            self.hid_buf = np.zeros((size, cell_size), dtype=np.float32)
        else:
            self.hid_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.epoch_idx = 0

    def store(self, obs, act, rew, val, logp, hid=0):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.hid_buf[self.ptr] = hid
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # if self.epoch_idx > 20:
        #     import pdb; pdb.set_trace()
        # print(rews)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, hid=self.hid_buf)
        self.epoch_idx += 1
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, beta=0.01, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=3e-4, train_pi_iters=80, train_v_iters=80, lam=0.95, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, use_rnn=False, reward_factor=1, spectrum_repr=False):
    """
    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(ac)

    # udpate env config
    # env.scalar_thick = ac_kwargs['scalar_thick']
    env.update_with_ac(**ac_kwargs)

    # For Tuple spaces
    obs_dim = ac.obs_dim

    if isinstance(env.action_space, spaces.Tuple):
        act_dim = core.tuple_space_dim(env.action_space, action=True)
    else:
        act_dim = env.action_space.shape

    # Create actor-critic module
    
    # print(ac)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, cell_size=ac_kwargs['cell_size'])

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):


        obs, act, adv, logp_old, hid = data['obs'], data['act'], data['adv'], data['logp'], data['hid']

        # for i in range(len(obs)-1):
        #     if torch.eq(obs[i], torch.zeros(12)).sum()==12 and torch.eq(obs[i+1], torch.zeros(12)).sum()==12:
        #         print(obs[i], obs[i+1], act[i], act[i+1])

        # Policy loss
        pis = []
        logp = 0

        if len(ac.pi) > 1: # tuple actions
            for i, actor_i in enumerate(ac.pi):
                pi, logp_i = actor_i(obs, act[:,i][:,None])
                logp += logp_i
                pis.append(pi)
        else:
                pi, logp_i = ac.pi[0](obs, act)
                logp += logp_i
                pis.append(pi)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        # sample estimation policy KL
        approx_kl = (logp_old - logp).mean().item()
        ent = sum([pi.entropy().mean().item() for pi in pis])
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return 0.5 * ((ac.v(obs) - ret)**2).mean()

    def compute_loss_pi_v_rnn(data):

        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']

        hid = torch.zeros(ac_kwargs['cell_size'])
        v = []
        logp = []
        ent = []
        num_traj = 0
        #todo: test
        for i in range(len(obs)):
            v_i, logp_i, hid, ent_i = ac.evaluate(obs[i], act[i], hid)
            if i < len(obs)-1 and obs[i+1].sum() == 0:
                num_traj += 1
                # print('Reinitialize #{}'.format(num_traj), flush=True)
                hid = torch.zeros(ac_kwargs['cell_size'])
            v.append(v_i)
            logp.append(logp_i)
            ent.append(ent_i)
        
        logp = torch.cat(logp)
        v = torch.cat(v)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # print(logp_old - logp)
        
        approx_kl = (logp_old - logp).mean().item()
        ent = torch.stack(ent).mean()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        loss_v = 0.5 * ((v - ret)**2).mean()
        # import pdb; pdb.set_trace()
        
        loss_pi = loss_pi - beta * ent

        logger.store(RetBuf=ret.clone().detach().numpy())
        # import pdb; pdb.set_trace()

        return loss_pi, pi_info, loss_v

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    if use_rnn:
        optimizer = Adam(ac.parameters(), lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # import pdb; pdb.set_trace()

        if not use_rnn:
            pi_l_old, pi_info_old = compute_loss_pi(data)
            v_l_old = compute_loss_v(data).item()

            #  Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                kl = mpi_avg(pi_info['kl'])
                if kl > 1.5 * target_kl:
                    logger.log(
                        'Early stopping at step %d due to reaching max kl.' % i)
                    break
                loss_pi.backward()
                mpi_avg_grads(ac.pi)    # average grads across MPI processes
                pi_optimizer.step()

            logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                vf_optimizer.zero_grad()
                if not use_rnn:
                    loss_v = compute_loss_v(data)
                loss_v.backward()
                mpi_avg_grads(ac.v)    # average grads across MPI processes
                vf_optimizer.step()

        else:
            pi_l_old, pi_info_old, v_l_old = compute_loss_pi_v_rnn(data)
            pi_l_old = pi_l_old.item()

            for i in range(train_pi_iters):
                optimizer.zero_grad()
                loss_pi, pi_info, loss_v = compute_loss_pi_v_rnn(data)
                kl = mpi_avg(pi_info['kl'])
                if kl > 1.5 * target_kl:
                    logger.log(
                        'Early stopping at step %d due to reaching max kl.' % i)
                    break
                loss = loss_pi + loss_v
                loss.backward()
                mpi_avg_grads(ac)
                optimizer.step()
            logger.store(StopIter=i)
        
        # import pdb; pdb.set_trace()
        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # import pdb; pdb.set_trace()
    # if ac_kwargs['scalar_thick']:
    #     thick= obs[env.num_materials:env.num_materials+env.num_thicknesses].argmax() / env.num_thicknesses
    #     obs = np.concatenate((obs[:env.num_materials+1], np.array([thick])))

    #                 if ac_kwargs['scalar_thick']:
    #             thick= obs[env.num_materials:env.num_materials+env.num_thicknesses].argmax() / env.num_thicknesses
    #             obs = np.concatenate((obs[:env.num_materials+1], np.array([thick])))
    hid = np.zeros(ac_kwargs['cell_size']) if ac_kwargs['cell_size'] else np.zeros(1)
    # import pdb; pdb.set_trace()

    design_tracker = DesignTracker(epochs, **logger_kwargs)
    total_env_time = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for t in range(local_steps_per_epoch):

            #TODO: only evaluate  
            act, v, logp, hid = ac.step(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(hid, dtype=torch.float32))

            # nv_start = time.time()
            next_obs, r, d, _ = env.step(act)
            # env_end = time.time()
            # env_time = env_end - env_start
            # total_env_time += env_time

            r = r * reward_factor # scale the rewards, possibly match the reward scale of atari
            ep_ret += r
            if not d:
                ep_len += 1

            # save and log
            if use_rnn:
                buf.store(obs, act, r, v, logp, hid)
            else:
                buf.store(obs, act, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1

            if terminal or epoch_ended:
                # print(t)
                # if epoch_ended and not(terminal):
                #     print('Warning: trajectory cut off by epoch at %d steps.'
                #           % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                # if timeout or epoch_ended:
                if not terminal:
                    _, v, _, _ = ac.step(
                        torch.as_tensor(obs, dtype=torch.float32),
                        torch.as_tensor(hid, dtype=torch.float32))
                else:
                    v = 0
                
                buf.finish_path(v)
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if hasattr(env, 'layers') and hasattr(env, 'thicknesses'):
                        design_tracker.store(env.layers, env.thicknesses, ep_ret, epoch)
                    
                        if rank == 0:
                            print(env.layers, env.thicknesses)

                obs, ep_ret, ep_len = env.reset(), 0, 0
                # reinitilize hidden state
                hid = np.zeros(ac_kwargs['cell_size'])
                if hasattr(env, "layers"):
                    logger.store(Act=act[1])

        # Save model
        
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
            design_tracker.save_state()

        # Perform PPO update!
        update()

        elapsed = time.time()-start_time
        epoch_time = time.time() - epoch_start_time
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        if hasattr(env, 'layers'):
            logger.log_tabular('Act', with_min_and_max=True)
        logger.log_tabular('RetBuf', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', elapsed)
        logger.log_tabular('FPS', int(steps_per_epoch/epoch_time))
        logger.dump_tabular()

    
    # print('Total env time {}'.format(total_env_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PerfectAbsorberVisNIR-v0')
    parser.add_argument('--discrete_thick', action='store_true',
                        help='whether to use discrete thickness')
    parser.add_argument('--use_rnn', action='store_true',
                        help='whether to use RNN')
    parser.add_argument('--cell_size', type=int, default=64,
                        help='GRU or LSTM cell size')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=0.01,
                        help='entropy loss coefficient')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--k_pi', type=int, default=4)
    parser.add_argument('--k_v', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--not_repeat', action='store_true',
                        help='if set to true, then do not take the same action for two consecutive steps')
    parser.add_argument('--hierarchical', action='store_true', help='if set to true, then output out the material type first, then condition the material thickness on the material type')
    parser.add_argument('--ortho_init', type=str, default='on',
                        help='if set to true, then use orthogonal initializaiton')
    parser.add_argument('--reward_factor', type=int, default=1, help='scaling factor for the env reward')
    parser.add_argument('--spectrum_repr', action='store_true', help='use spectrum')
    parser.add_argument('--scalar_thick', action='store_true', help='whether to use a single sclalr to represent thickness')
    parser.add_argument('--bottom_up', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # logger_kwargs

    # import pdb; pdb.set_trace()

    env_kwargs = {"discrete_thick":args.discrete_thick, 'spectrum_repr':args.spectrum_repr, 'bottom_up':False}
    ppo(get_env_fn(args.env, **env_kwargs),
        actor_critic=core.MLPActorCritic if not args.use_rnn
        else core.RNNActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l,
                       cell_size=args.cell_size if args.use_rnn else None,
                       not_repeat=args.not_repeat,
                       ortho_init=args.ortho_init,
                       hierarchical=args.hierarchical,
                       channels=args.channels,
                       scalar_thick=args.scalar_thick),
        lam = args.lam,
        use_rnn=args.use_rnn,
        gamma=args.gamma,
        beta=args.beta,
        pi_lr=args.lr,
        vf_lr=args.lr,
        seed=args.seed,
        steps_per_epoch=args.steps,
        max_ep_len=args.max_ep_len,
        train_pi_iters=args.k_pi,
        train_v_iters=args.k_v,
        clip_ratio=args.ratio,
        epochs=args.epochs,
        reward_factor=args.reward_factor,
        logger_kwargs=logger_kwargs,
        spectrum_repr=args.spectrum_repr,
        target_kl=1e-3)
