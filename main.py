from expconfigs import get_runner
from RLMultilayer.algos.ppo.ppo import ppo

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--exp', type=str, default='pa_compare')
    parser.add_argument('--name', type=str, default='test')
    args = parser.parse_args()

    eg = get_runner(args)
    eg.run(ppo, num_cpu=args.cpu, data_dir='/home/hzwang/Experiments/{}'.format(args.name), datestamp=False)