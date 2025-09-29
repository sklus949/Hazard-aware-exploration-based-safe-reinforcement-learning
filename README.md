# CrowdNav


## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.

Note: If you are training and testing on a server, please add the following lines at the beginning of your script.
```
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
```
1. Train a policy.
```
python train.py --policy hesrl
```
2. Test policy with 500 test cases.
```
python test.py --policy hesrl --phase test
```

3. Plot training curve.
```
python utils/plot.py data/hesrl/output.log
```
4. Visualize by adding the following line in the code.
```
env.render(mode="video")
```