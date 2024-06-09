import json
import numpy as np
from os.path import join
import pdb
import os

from diffuser.guides.policies import TATPolicy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.tree.tree import TrajAggTree


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)
num_eval = 1000
seed = 0

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

#--------------------------------- policy -------------------------------#
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


policy = TATPolicy(diffusion, dataset.normalizer, use_tree = args.use_tree)


#---------------------------------- main loop ----------------------------------#

for _ in range(num_eval):
    seed += 1
    env = datasets.load_environment(args.dataset)
    env.seed(seed)

    if args.use_tree:
        traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                     traj_dim=observation_dim,
                                     one_minus_alpha=args.one_minus_alpha,
                                    )
        policy.reset_tree(traj_agg_tree)
        print(f"Seed ({seed}), TAT planning") 
    else:
        print(f"Seed ({seed}), Vanllia planning")

    savepath_i = os.path.join(args.savepath, str(seed))
    if not os.path.exists(savepath_i): 
        os.mkdir(savepath_i)

    observation = env.reset()

    if args.multi_task:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation
            cond_draw = cond.copy()

            _, samples = policy(cond, batch_size=args.batch_size)
            tree_observations_render = samples.observations[0].copy()
            # actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action}'
        )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(savepath_i, f'{t}.png')

            if t == 0: 
                # print first 10 plans sampled from vanilla diffuser
                for k in range(min(10, len(samples.observations_render))):
                    renderer.composite(join(savepath_i, f'diffuser_plan{k}.png'), samples.observations_render[k][None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)
                    # renderer.composite(fullpath, samples.observations_render[:4], ncol=1)
                if args.use_tree:
                    renderer.composite(join(savepath_i, f'tree_plan.png'), np.array(tree_observations_render)[None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)


            # renderer.render_plan(join(savepath_i, f'{t}_plan.mp4'), samples.actions, samples.observations, state)
            ## save rollout thus far
            renderer.composite(join(savepath_i, 'rollout.png'), np.array(rollout)[None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)

            # renderer.render_rollout(join(savepath_i, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(savepath_i, f'rollout.mp4'), plan=join(savepath_i, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(savepath_i, 'rollout.json')
    json_data = {'seed': seed, 'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

    env.close()
    del env
    if args.use_tree:
        del traj_agg_tree
