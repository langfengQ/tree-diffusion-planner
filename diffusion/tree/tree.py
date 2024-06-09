import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def cosine_similarity(x, y):
    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return similarity

def get_weight(step, tree_lambda):
    return np.power(tree_lambda, step-1) 

class TreeNode(object):
    '''
    A node in the TAT.
    Each node keeps track of its own state, visiting states, weights, and step (for debug)
    '''

    def __init__(self, parent, state, tree_lambda=0.99):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._tree_lambda = tree_lambda
        self._states = [state['state']]
        self._steps = [state['step']]
        self._weights = [state['weight']]
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights)+1e-10)

    
    @property
    def _num_children(self):
        return len(self._children)

    def expand(self, state):
        '''
        Expand tree by creating new children.
        '''
        self._children[self._num_children] = TreeNode(self, state, self._tree_lambda)
        return self._children[self._num_children - 1]


    def is_children(self, state, dis_threshold):
        '''
        Find most suitable node to transition.
        '''
        min_distance = 9999
        min_distance_key = None
        for key in self._children.keys():
            node_state = self._children[key].node_state
            distance = cosine_similarity(state, node_state)
            distance = 1 - distance
            if distance < min_distance:
                min_distance = distance
                min_distance_key = key
        if min_distance < dis_threshold:
            return True, min_distance_key
        else:
            return False, None
            
            
    def update_children(self, state, key):
        '''
        Update the statistics of this node.
        '''
        self._children[key]._states.append(state['state'])
        self._children[key]._steps.append(state['step'])
        self._children[key]._weights.append(state['weight'])
        self._children[key].update_node_state()


    def update_node_state(self,):
        '''
        Update the state of this node.
        '''
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights))


    def get_value(self):
        '''
        Return the total weights for this node.
        '''
        return np.sum(np.array(self._weights))


    def step(self, key):
        '''
        Transition to a specific child node.
        '''
        return self._children[key]


    def is_leaf(self):
        '''
        check if leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None


class TrajAggTree(object):
    '''
    An implementation of Trajectory Aggregation Tree (TAT).
    '''
    def __init__(self, tree_lambda, traj_dim, action_dim=None, one_minus_alpha=0.005, start_state=None):
        self._tree_lambda = tree_lambda
        self._distance_threshold = one_minus_alpha # 1-\alpha
        self.traj_dim = traj_dim
        self.action_dim = action_dim
        if start_state is None:
            start_state = np.zeros((traj_dim,))
        state = {'state': start_state, 'step': 0, 'weight': 1}
        self._root = TreeNode(None, state, self._tree_lambda)


    def integrate_single_traj(self, traj, length, history_length):
        '''
        Integrate a single trajectory into the tree.
        '''
        node = self._root

        # Merging the former sub-trajectory
        for i in range(history_length, length):
            if node.is_leaf():
                break
            is_children, key = node.is_children(traj[i], self._distance_threshold)

            if is_children:
                state = {'state': traj[i], 'step': i, 'weight': get_weight(i, self._tree_lambda)} 
                node.update_children(state, key=key)
                node = node.step(key)
            else:
                # no suitable nodes for transition
                break
        
        # Expanding the latter sub-trajectory
        if i < length - 1:
            for j in range(i, length):
                state = {'state': traj[j], 'step': j, 'weight': get_weight(j, self._tree_lambda)}
                node = node.expand(state)


    def integrate_trajectories(self, trajectories, history_length=1):
        '''
        Integrate a batch of new trajectories sampled from diffusion planners.
        history_length: trajectories contain historical states (e.g., one history state in Diffuser). We will not integrate the historical part.
        '''
        assert len(trajectories.shape) == 3 and trajectories.shape[-1] == self.traj_dim

        batch_size, length = trajectories.shape[0], trajectories.shape[1]

        for i in range(batch_size):
            self.integrate_single_traj(trajectories[i], length, history_length)



    def get_next_state(self,):
        '''
        Acting: select the most impactful node, which has highest weight among the child nodes.
        '''
        selected_key, node = max(self._root._children.items(), key=lambda node: node[1].get_value())
        visit_time = len(node._states)
        max_depth = np.array(node._steps).max()
        return node.node_state, selected_key, visit_time, max_depth
        

    def pruning(self, selected_key):
        '''
        Pruning: prune the tree, keeping in sync with the environment.
        '''
        self._root = self._root._children[selected_key]
        self._root._parent = None


    def forward_state(self, trajectories, action_dim=None, first_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = trajectories[:, :, :self.action_dim]
            if first_action is None:
                first_action = np.zeros_like(_actions)[:, 0, :][:, None, :]
            _actions = _actions[:,:-1,:] # discard the last action
            _actions = np.concatenate([first_action, _actions], axis=1)
            _observations = trajectories[:, :, self.action_dim:]
            tree_trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            tree_trajectories = trajectories
        return tree_trajectories
    
    def reverse_state(self, tree_trajectories, action_dim=None, last_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = tree_trajectories[:, :, :self.action_dim]
            if last_action is None:
                # pad with the current last action
                last_action = _actions[:, -1, :].copy()
                last_action = last_action[:, None, :]
            _actions = _actions[:, 1:, :] # discard the first action
            _actions = np.concatenate([_actions, last_action], axis=1)
            _observations = tree_trajectories[:, :, self.action_dim:]
            trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            trajectories = tree_trajectories
        return trajectories

    def __str__(self):
        return "TrajAggTree"