#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import IPython
import numpy as np

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

#from bag_deep_ckt.autockt.envs.bag_opamp_discrete import TwoStageAmp
from envs.spectre_vanilla_opamp import TwoStageAmp

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))
register_env("opamp-v0", lambda config:TwoStageAmp(config))

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    parser.add_argument(
        "--num_val_specs",
        type=int,
        default=50,
        help="Number of untrained objectives to test on")
    parser.add_argument(
        "--traj_len",
        type=int,
        default=60,
        help="Length of each trajectory")
    return parser

def run(args, parser):
    config = args.config
    if not config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = 0#min(2, config["num_workers"])

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout(agent, args.env, num_steps, args.out, args.no_render)

def unlookup(norm_spec, goal_spec):
    spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1) 
    return spec

def rollout(agent, env_name, num_steps, out="assdf", no_render=True):
    if hasattr(agent, "local_evaluator"):
        #env = agent.local_evaluator.env
        env_config = {"generalize":True,"num_valid":args.num_val_specs, "save_specs":False, "run_valid":True}
        if env_name == "opamp-v0":
            env = TwoStageAmp(env_config=env_config)
    else:
        env = gym.make(env_name)

    #get unnormlaized specs
    norm_spec_ref = env.global_g
    spec_num = len(env.specs)
     
    if hasattr(agent, "local_evaluator"):
        state_init = agent.local_evaluator.policy_map[
            "default"].get_initial_state()
    else:
        state_init = []
    if state_init:
        use_lstm = True
    else:
        use_lstm = False

    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_array = []
    action_arr_comp = []
    rollout_steps = 0
    reached_spec = 0
    while rollout_steps < args.num_val_specs:
        if out is not None:
            rollout_num = []
        state = env.reset()
        
        done = False
        reward_total = 0.0
        steps=0
        while not done and steps < args.traj_len:
            if use_lstm:
                action, state_init, logits = agent.compute_action(
                    state, state=state_init)
            else:
                action = agent.compute_action(state)
                action_array.append(action)

            next_state, reward, done, _ = env.step(action)
            print(action)
            print(reward)
            print(done)
            reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout_num.append(reward)
                next_states.append(next_state)
            steps += 1
            state = next_state
        norm_ideal_spec = state[spec_num:spec_num+spec_num]
        ideal_spec = unlookup(norm_ideal_spec, norm_spec_ref)
        if done == True:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
            action_array = []
            pickle.dump(action_arr_comp, open("action_arr_test", "wb"))
        else:
            obs_nreached.append(ideal_spec)          #save unreached observation 
            action_array=[]
        if out is not None:
            rollouts.append(rollout_num)
        print("Episode reward", reward_total)
        rollout_steps+=1
        #if out is not None:
            #pickle.dump(rollouts, open(str(out)+'reward', "wb"))
        pickle.dump(obs_reached, open("opamp_obs_reached_test","wb"))
        pickle.dump(obs_nreached, open("opamp_obs_nreached_test","wb"))
        print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached))) 

    print("Num specs reached: " + str(reached_spec) + "/" + str(args.num_val_specs))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
