import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from bag_deep_ckt.autockt.envs.spectre_vanilla_opamp_45nm import TwoStageAmp
from bag_deep_ckt.autockt.envs.spectre_fc import FoldedCascode 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args.add_argument('--env', '-env', type=str)
args = parser.parse_args()
ray.init()

config_validation = {
            "sample_batch_size": 200,
            "train_batch_size": 1200,
            "sgd_minibatch_size": 1200,
            "num_sgd_iter":3,
            "lr":1e-3,
            "vf_loss_coeff":0.5,
            "horizon":  60,#tune.grid_search([15,25]),
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 7,
            "env_config":{"generalize":False, "save_specs":False, "run_valid":True},
            }

config_train = {
            "sample_batch_size": 60,#200,
            "train_batch_size": 360,
            "sgd_minibatch_size": 360,
            "num_sgd_iter": 3,
            "lr":1e-3,
            "vf_loss_coeff": 0.5,
            "horizon":  60,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 8,
            "env_config":{"generalize":True, "save_specs":True, "run_valid":False},
            }

if not args.checkpoint_dir:
    trials = tune.run_experiments({
        "train_45nm": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": -0.02},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    #print(trials[0]._checkpoint.value)
    tune.run_experiments({
        "restore_ppo": {
        "run": "PPO",
        "config": config_validation,
        "env": TwoStageAmp,
        #"restore": trials[0]._checkpoint.value},
        "restore": args.checkpoint_dir,
        "checkpoint_freq":2},
    })
