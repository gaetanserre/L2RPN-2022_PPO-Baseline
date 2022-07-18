## L2RPN WCCI 2022 PPO Baseline

This repository contains the code to create and train the Proximal Policy Optimization baseline
provided in the [2022 edition of Learning to Run a Power Network](https://codalab.lisn.upsaclay.fr/competitions/5410).


### Usage
You need python>=3.8.6

To train the baseline:
```bash
pip[3] install -r requirements.txt
python[3] train.py [args]
```
Run `python[3] train.py -h` to see all available arguments.

To reproduce the figures (6), (7) and (8) of 
[Reinforcement learning for Energies of the future and carbon neutrality: a Challenge Design](),
check the corresponding `make_figure_n.ipynb` notebook.

You need to train several instances of the baseline agent before
running these notebooks.

### Work in progress
We used this code to generate the results of
[Reinforcement learning for Energies of the future and carbon neutrality: a Challenge Design]() but we are still working on cleaning it up.
