# OML-PPO

PyTorch implementation for OML-PPO (https://arxiv.org/abs/2006.11940)

## Prerequisites
- python=3.7.4
- torch=1.3.1
- torchvision=0.5.0
- tmm=0.1.7
- spinningup=0.2.0

## Run experiments

Max length = 6:  
python ppo_absorber_visnir.py --cpu 16 --maxlen 6 --exp_name absorber6 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v0

Max length = 15:  
python ppo_absorber_visnir.py --cpu 16 --maxlen 15 --exp_name perfect_absorber15 --use_rnn --discrete_thick --num_runs 1 --env PerfectAbsorberVisNIR-v1

## Plotting results
Use final_results.ipynb to plot the results.
