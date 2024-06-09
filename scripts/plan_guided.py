import pdb
import os

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.tree.tree import TrajAggTree

import numpy as np
import gym


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema

# if warm start
assert (args.vanilla_warm_start and args.tree_warm_start)==False, "'vanilla_warm_start' and 'tree_warm_start' cannot be True at the same time"
if args.tree_warm_start:
    assert args.use_tree, "'tree_warm_start' is True, but 'use_tree' is False"
is_warm_start = args.vanilla_warm_start or args.tree_warm_start
diffusion.is_warm_start = is_warm_start
diffusion.warm_start_step = args.warm_start_step
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## tree
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_tree = args.use_tree,
    tree_warm_start = args.tree_warm_start,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

num_eval = 50
seed = 0

sampling_times = []
tree_times = []
other_times = []
total_times = []
observation_old = None

for _ in range(num_eval):
    seed += 1
    env = gym.make(args.dataset)
    env.seed(seed)
    if args.use_tree:
        traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                     traj_dim=observation_dim+action_dim,
                                     action_dim=action_dim,
                                     one_minus_alpha=args.one_minus_alpha,
                                    )
        policy.reset_tree(traj_agg_tree)
        print(f"Seed ({seed}), TAT planning") 
    else:
        print(f"Seed ({seed}), Vanllia planning")

    savepath_i = os.path.join(args.savepath, str(seed))
    logger.set_savepath(savepath_i)
    observation = env.reset()
    if observation_old is not None:
        assert np.all(observation_old == observation)==False, "observation is the same"
    else:
        observation_old = observation
    ## observations for rendering
    rollout = [observation.copy()]


    total_reward = 0
    policy.diffusion_model.reset_warm_start()

    for t in range(args.max_episode_length):
        if t % 10 == 0: print(savepath_i, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples, _, _ = policy(conditions, step=t, batch_size=args.batch_size, verbose=args.verbose, tree_batch_size=args.tree_batch_size)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        if t % 10 == 0:
            print(
                f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                f'values: {samples.values} | scale: {args.scale}',
                flush=True,
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples=samples)

        if terminal:
            break

        observation = next_observation

    if args.is_render:
        logger.log(t, rollout=rollout)
    ## write results to json file at `savepath_i`
    logger.finish(t, np.array(rollout), score, total_reward, terminal, diffusion_experiment, value_experiment)
    if args.use_tree:
        del traj_agg_tree
    env.close()
    del env
