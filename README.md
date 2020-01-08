# AutoCkt
Code for [Deep Reinforcement Learning of Analog Circuit Designs](https://arxiv.org/), presented at Design Automation and Test in Europe, 2020. Note that the results shown in the paper include those from NGSpice and Spectre. NGSpice is free and can be installed online (see Setup section). Spectre requires a license, as well as access to the particular technology; the code for this will be open sourced at a later time.

## Setup
This setup requires Anaconda. In order to obtain the required packages, run the command below from the top level directory of the repo to install the Anaconda environment:

```
conda env create -f environment.yml
```

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
Make sure that you are in the Anaconda environment. From the top level directory run:

```
python autockt/val_autobag_ray.py
```
The training checkpoints will be saved in your home directory under ray\_results. Tensorboard can be used to automatically load results using the command:

```
tensorboard --logdir path/to/checkpoint
```
