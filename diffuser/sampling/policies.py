from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np

Trajectories = namedtuple('Trajectories', 'actions observations values')

import time
import copy

class GuidedPolicy:
    """
    Vanilla diffuser policy from https://github.com/jannerm/diffuser/blob/main/diffuser/sampling/policies.py.
    """
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

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


class TATGuidedPolicy(GuidedPolicy):
    """
    Policy for TAT
    """

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, use_tree, tree_warm_start, **sample_kwargs):
        self.use_tree = use_tree
        self.tree = None
        self.tree_warm_start = tree_warm_start
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


    def __call__(self, conditions, step, batch_size=1, verbose=True, tree_batch_size=None):
        if self.use_tree:
            return self.tat_call(conditions, step, batch_size=batch_size, verbose=verbose, tree_batch_size=tree_batch_size)
        else:
            action, trajectories = super().__call__(conditions, batch_size, verbose=verbose)
            return action, trajectories, None, None


    def tat_call(self, conditions, step, batch_size=1, verbose=True, tree_batch_size=None):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        if self.tree_warm_start and step != 0:
            assert self.use_tree == True, "trajectory aggregation tree is not used"
            if self.tree is None:
                raise ValueError("trajectory aggregation tree is None")
            else:
                self.x0_warm_start_from_tree(batch_size)

        # Sample plans via vanilla diffuser.
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        # Preprocessing
        tree_trajectories = self.tree.forward_state(trajectories, action_dim=self.action_dim)
        
        # Merging + Expanding
        self.tree.integrate_trajectories(tree_trajectories[:tree_batch_size,:,:])

        # Acting 
        next_sample, selected_key, visit_time, max_depth = self.tree.get_next_state()
        action = next_sample[:self.action_dim]
        action = self.normalizer.unnormalize(action, 'actions')
        
        # Pruning 
        self.tree.pruning(selected_key)

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)

        return action, trajectories, visit_time, max_depth
    
    
    def reset_tree(self, traj_agg_tree):
        if self.tree is not None:
            del self.tree
        self.tree = traj_agg_tree

    def x0_warm_start_from_tree(self, batch_size):
        '''
        Rollout x0 from the tree for warm start planning.
        '''
        rollout_tree = copy.deepcopy(self.tree)
        x0 = []
        x0.append(rollout_tree._root.node_state)
        terminal = rollout_tree._root.is_leaf()
        while terminal == False:
            next_sample, selected_key, _, _ = rollout_tree.get_next_state()
            x0.append(next_sample)
            rollout_tree.pruning(selected_key)
            terminal = rollout_tree._root.is_leaf()

        x0 = np.repeat(np.array(x0)[None], batch_size, axis=0)
        x0_pad = utils.to_np(self.diffusion_model.x0_warm_start)
        pad_length = self.diffusion_model.horizon - x0.shape[1]
        assert pad_length > 0
        last_action = x0_pad[:, -(pad_length+1), :self.action_dim][:, None, :]
        x0 = rollout_tree.reverse_state(x0, action_dim=self.action_dim, last_action=last_action)
        x0_warm_start = np.concatenate([x0, x0_pad[:, -pad_length:, :]], axis=1)
        x0_warm_start = utils.to_torch(x0_warm_start, dtype=torch.float32, device='cuda:0')
        
        self.diffusion_model.x0_warm_start = x0_warm_start

        del rollout_tree