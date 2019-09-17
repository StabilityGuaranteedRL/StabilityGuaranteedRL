# StabilityGuaranteedRL
## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.


To create a conda env with python3, one runs 
```bash
conda create -n test python=3.6
```
To activate the env: 
```
conda activate test
```

# Installation Environment

```bash
https://github.com/StabilityGuaranteedRL/StabilityGuaranteedRL.git
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib

```


### Example 1. Training of LAC
```
python main.py
```
Specify the environment for evaluation in VARIANT{'env_name'} and algorithm in VARIANT{'algorithm_name'}.
The algorithm_name could be one of ['LAC','SAC_cost']
The hyperparameters could be adjested in ALG_PARAMS.

Other hyperparameter are also ajustable in variant.py.
```bash
VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'Antcost-v0',
    'env_name': 'oscillator',
    # 'env_name': 'HalfCheetahcost-v0',
    # 'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SAC_cost',
    # 'additional_description': '-horizon=inf-weight-4-10-0-0-0',
    # 'additional_description': '-new',
    # 'additional_description': '-horizon=inf-scale-cost=0.01-gamma=0.75-maxa=1.-1e6',
    'additional_description': '',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    #evaluation params
    # 'evaluation_form': 'constant_impulse',
    'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # cartpole
        # 'LAC',
        # 'SAC',
        # 'LQR',
        'SAC_cost-new',
        # halfcheetah
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'LAC-des=1-horizon=inf',
        # 'SAC',

        # ant
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'SAC_cost-des=1-no_contrl_cost',

        # Fetch
        # 'LAC',
        # 'SAC',
        # 'SAC_cost-0.75-new',
        #oscillator
        # 'LAC',
        # 'SAC',
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2048,
}
```

### Example 2. Evaluation of Robustness
Run the following in bash after the training has completed.
```
python robustness_eval
```

Specify the environment for evaluation in VARIANT{'env_name'} and list of agents to evaluate in VARIANT{'eval_list'}.
The form of evaluation is specified in VARIANT{'evaluation_form'}, with hyperparameters adjusted in EVAL_PARAMS.



Other hyperparameter are also ajustable in variant.py.
```bash
VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'Antcost-v0',
    'env_name': 'oscillator',
    # 'env_name': 'HalfCheetahcost-v0',
    # 'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SAC_cost',
    # 'additional_description': '-horizon=inf-weight-4-10-0-0-0',
    # 'additional_description': '-new',
    # 'additional_description': '-horizon=inf-scale-cost=0.01-gamma=0.75-maxa=1.-1e6',
    'additional_description': '',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    #evaluation params
    # 'evaluation_form': 'constant_impulse',
    'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # cartpole
        # 'LAC',
        # 'SAC',
        # 'LQR',
        'SAC_cost-new',
        # halfcheetah
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'LAC-des=1-horizon=inf',
        # 'SAC',

        # ant
        # 'LAC-des=1-horizon=inf-alpha=1',
        # 'SAC_cost-des=1-no_contrl_cost',

        # Fetch
        # 'LAC',
        # 'SAC',
        # 'SAC_cost-0.75-new',
        #oscillator
        # 'LAC',
        # 'SAC',
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2048,
}
