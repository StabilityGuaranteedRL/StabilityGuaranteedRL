import gym
import datetime
import numpy as np
import ENV.env
SEED = None

VARIANT = {
    # 'env_name': 'FetchReach-v1',
    # 'env_name': 'Antcost-v0',
    'env_name': 'oscillator',
    # 'env_name': 'oscillator_complicated',
    # 'env_name': 'HalfCheetahcost-v0',
    # 'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SAC_cost',
    'additional_description': '',
    # 'additional_description': '-new',
    # 'additional_description': '-horizon=inf-scale-cost=0.01-gamma=0.75-maxa=1.-1e6',
    # 'additional_description': '',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    #evaluation params
    'evaluation_form': 'constant_impulse',
    # 'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    # 'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # cartpole
        'LAC',
        'SAC',
        # 'LAC-horizon=5-quadratic',
        # 'LQR',
        # 'SAC_cost-new',
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
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
    'oscillator': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,},
    'oscillator_complicated': {
        'max_ep_steps': 400,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,},
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': False,},
    'Quadrotorcost-v0': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Antcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 8,
        'eval_render': False,},
    'FetchReach-v1': {
        # 'max_ep_steps': 50,
        'max_ep_steps': 200,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'disturbance dim': 4,
        'eval_render': False, },
}
ALG_PARAMS = {
    'MPC':{
        'horizon': 5,
    },

    'LQR':{
        'use_Kalman': False,
    },

    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'alpha3': 1.,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 5,
        'finite_horizon': True,
        'soft_predict_horizon': False,
        # 'finite_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },


    'SAC_cost': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None,

    },
}


EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 10),
        'magnitude_range': np.arange(0.1, 1.1, .1),
        'num_of_paths': 100,   # number of path for evaluation
        'impulse_instant': 100,
    },
    'constant_impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        'magnitude_range': np.arange(0.2, 2.2, .2),
        # 'magnitude_range': np.arange(0.1, 1.0, .1),
        'num_of_paths': 100,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([80]),
        # 'grid_eval': False,
        'num_of_paths': 100,   # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'path': './log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/',
        'num_of_paths': 100,   # number of path for evaluation
    },
    'dynamic': {
        'additional_description': 'original',
        'num_of_paths': 10,   # number of path for evaluation
        'plot_average': True,
        'directly_show': True,
    },
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = True
def get_env_from_name(name):
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_v2':
        from envs.ENV_V2 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_partial':
        from envs.ENV_V3 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real':
        from envs.ENV_V4 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_swing_up':
        from envs.ENV_V5 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real_no_friction':
        from envs.ENV_V6 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_motor':
        from envs.ENV_V7 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_fitted_motor':
        from envs.ENV_V8 import CartPoleEnv_adv as dreamer
        env = dreamer(eval=True)
        env = env.unwrapped
    elif name == 'oscillator':
        from envs.oscillator import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_complicated':
        from envs.oscillator_complicated import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'Quadrotorcost-v0':
        env = gym.make('Quadrotorcons-v0')
        env = env.unwrapped
        env.modify_action_scale = False
        env.use_cost = True

    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotorcons-v0':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    env.seed(SEED)
    return env

def get_train(name):
    if 'RARL' in name:
        from LAC.RARL import train as train
    elif 'LAC' in name:
        from LAC.LAC_V1 import train
    else:
        from LAC.SAC_cost import train

    return train

def get_policy(name):
    if 'RARL' in name:
        from LAC.RARL import RARL as build_func
    elif 'LAC' in name :
        from LAC.LAC_V1 import LAC as build_func
    elif 'LQR' in name:
        from LAC.lqr import LQR as build_func
    elif 'MPC' in name:
        from LAC.MPC import MPC as build_func
    else:
        from LAC.SAC_cost import SAC_cost as build_func
    return build_func

def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval


