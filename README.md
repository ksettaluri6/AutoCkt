UPDATE: please see the journal paper for additional results on designing a two-stage folded-cascode and two-stage amplifier (https://ieeexplore.ieee.org/abstract/document/9576505). 

# AutoCkt: Deep Reinforcement Learning of Analog Circuit Designs
Code for [Deep Reinforcement Learning of Analog Circuit Designs](https://arxiv.org/abs/2001.01808), presented at Design Automation and Test in Europe, 2020. Note that the results shown in the paper include those from NGSpice and Spectre. NGSpice is free and can be installed online (see Setup). Spectre requires a license, as well as access to the particular technology; the code for this will be open sourced at a later time.

## Setup
This setup requires Anaconda. In order to obtain the required packages, run the command below from the top level directory of the repo to install the Anaconda environment:

```
source /cktgym-dev-disk1/miniconda3/bin/activate
conda create --prefix ~/dir/py35 python=3.5
conda activate /home/dir/symlink/py35
sh /scripts/install.h
```

You might need to install some packages further using pip if necessary. To ensure the right versions, look at the environment.yml file.

NGspice 2.7 needs to be installed separately, via this [installation link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/). Page 607 of the pdf manual on the website has instructions on how to install. Note that you might need to remove some of the flags to get it to install correctly for your machine. 

## Code Setup
The code is setup as follows:

<img src=readme_images/flowchart.png width="500">

The top level directory contains two sub-directories:
* AutoCkt/: contains all of the reinforcement code
    * val_autobag_ray.py: top level RL script, used to set hyperparameters and run training
    * rollout.py: used for validation of the trained agent, see file for how to run
    * envs/ directory: contains all OpenAI Gym environments. These function as the agent in the RL loop and contain information about parameter space, valid action steps and reward.
* eval\_engines/: Contains all of the code pertaining to simulators
    * ngspice/: this directory runs all NGSpice related scripts.
        * netlist_templates: the exported netlist file modified using Jinja to update any MOS parameters
        * specs_test/: a directory containing a unique yaml file for each circuit with information about design specifications, parameter ranges, and how to normalize. 
        * script_test/: directory with files that test functionality of interface scripts  

## Training AutoCkt
Make sure that you are in the Anaconda environment. Before running training, the circuit netlist must be modified in order to point to the right library files in your directory. To do this, run the following command:
```
python Ckt/scripts/correct_inputs.py 
```

To generate the design specifications that the agent trains on, run:
```
python Auto/autockt/gen_specs.py --num_specs ##
```
The result is a pickle file dumped to the gen_specs/ folder.

To train the agent, open ipython from the top level directory and then: 
```
run Auto/autockt/val_autobag_ray.py
```
The training checkpoints will be saved in your home directory under ray\_results. Tensorboard can be used to load reward and loss plots using the command:

```
tensorboard --logdir path/to/checkpoint
```

To replicate the results from the paper, num_specs 350 was used (only 50 were selected for each CPU worker). Ray parallelizes according to number of CPUs available, that affects training time. 
## Validating AutoCkt
The rollout script takes the trained agent and gives it new specs that the agent has never seen before. To generate new design specs, run the gen_specs.py file again with your desired number of specs to validate on. To run validation, open ipython:

```
run Auto/autockt/rollout.py /path/to/ray/checkpoint --run PPO --env opamp-v0 --num_val_specs ### --traj_len ## --no-render
``` 
* num_val_specs: the number of untrained objectives to test on
* traj_len: the length of each trajectory

Two pickle files will be updated: opamp_obs_reached_test and opamp_obs_nreached_test. These will contain all the reached and unreached specs, respectively.

## Results
Please note that results vary greatly based on random seed and spec generation (both for testing and validation). An example spec file is provided that was used to generate the results below. 

<img src=readme_images/results.png width="800">

The rollout generalization results will show up as pickle files opamp_obs_reached_test and opamp_obs_nreached_test. For this particular run, we obtained 938/1000. Additional runs were also conducted, and we found that the results varied from 80%-96% depending on the generated specs during rollout, and the specs that were changed during training. Our results were obtained by running on an 8 core machine, we've found that running on anything below 2 cores results in weird training behavior. 
