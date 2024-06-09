from collections import namedtuple
import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations observations_render')

class Policy:
    """
    Vanilla diffuser policy from https://github.com/jannerm/diffuser/blob/maze2d/diffuser/guides/policies.py.
    """

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations, observations)
        return action, trajectories
        # else:
        #     return action


class TATPolicy(Policy):
    """
    Policy for TAT
    """

    def __init__(self, diffusion_model, normalizer, use_tree):
        self.use_tree = use_tree
        self.tree = None
        super().__init__(diffusion_model, normalizer)


    def __call__(self, conditions, debug=False, batch_size=1):
        if self.use_tree:
            return self.tat_call(conditions, debug, batch_size)
        else:
            return super().__call__(conditions, debug, batch_size)


    def tat_call(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        # Sample plans via vanilla diffuser.
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample)

        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        state_of_sample = sample[:, :, self.action_dim:] # only take the observation
        planning_horizon = state_of_sample.shape[1]

        normed_observations = sample[:, :, self.action_dim:]
        observations_render = self.normalizer.unnormalize(normed_observations, 'observations')
        plan_of_tree = []

        # Merging + Expanding
        self.tree.integrate_trajectories(state_of_sample)

        # Get a plan via open-loop planning
        plan_of_tree.append(state_of_sample[0,0])
        for i in range(planning_horizon - 1):
            # Acting 
            next_sample, selected_key, _, _ = self.tree.get_next_state()
            plan_of_tree.append(next_sample)

            # Pruning 
            self.tree.pruning(selected_key)

        plan_of_tree = np.array(plan_of_tree)[None]
        observations = self.normalizer.unnormalize(plan_of_tree, 'observations')

        trajectories = Trajectories(None, observations, observations_render)
        return None, trajectories


    def reset_tree(self, traj_agg_tree):
        if self.tree is not None:
            del self.tree
        self.tree = traj_agg_tree