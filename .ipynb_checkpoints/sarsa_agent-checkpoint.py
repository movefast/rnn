import numpy as np
from agent import BaseAgent
from gwt import GridWorldTileCoder
from utils import argmax


# SARSA
class SarsaAgent(BaseAgent):
    """
    Initialization of Sarsa Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        
        # Create an array for action-value estimates and initialize it to zero.
        # self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.
        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        # self.w = np.ones((self.num_actions, self.num_states*6))
        self.w = np.ones((self.num_actions, agent_init_info['iht_size']))
        
        # We initialize self.mctc to the mountaincar verions of the 
        # tile coder that we created
        # self.tc = GridWorldTileCoder(iht_size=self.num_states*6, num_tilings=12, num_tiles=4)
        self.tc = GridWorldTileCoder(iht_size=agent_init_info['iht_size'], num_tilings=agent_init_info['num_tilings'], num_tiles=agent_init_info['num_tiles'])

    def select_action(self, tiles):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        action_values = []
        chosen_action = None
        
        # First loop through the weights of each action and populate action_values
        # with the action value for each action and tiles instance
        
        # Use np.random.random to decide if an exploritory action should be taken
        # and set chosen_action to a random action if it is
        # Otherwise choose the greedy action using the given argmax 
        # function and the action values (don't use numpy's armax)
        
        ### BEGIN SOLUTION
        for action in self.w:
            action_values.append(action[tiles].sum())

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)
        else:
            chosen_action = argmax(action_values)
        ### END SOLUTION
        
        return chosen_action, action_values[chosen_action]

    def get_state_feature(self, state):
        state, is_door = state
        if self.feature is None:
            self.feature = state, is_door
        else:
            # self.feature = self.feature[0] * 0.9 + state * 0.1, self.feature[1] * .9 + is_door * .1
            if is_door == True:
                self.feature = state, is_door
            else:
                # self.feature = state, self.feature[1] * .9 + is_door * .1
                self.feature = state, self.feature[1] * .1 + is_door * .9
        return self.feature

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.feature = None
        position, velocity = self.get_state_feature(state)
        
        # Use self.tc to set active_tiles using position and velocity
        # set current_action to the epsilon greedy chosen action using
        # the select_action function above with the active tiles
        
        ### BEGIN SOLUTION
        active_tiles = self.tc.get_tiles(position, velocity)
        
        current_action, _ = self.select_action(active_tiles)
        ### END SOLUTION
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        self.steps = 0
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # choose the action here
        position, velocity = self.get_state_feature(state)
        
        # Use self.tc to set active_tiles using position and velocity
        # set current_action and action_value to the epsilon greedy chosen action using
        # the select_action function above with the active tiles
        
        # Update self.w at self.previous_tiles and self.previous action
        # using the reward, action_value, self.discount, self.w,
        # self.step_size, and the Sarsa update from the textbook
        
        ### BEGIN SOLUTION
        active_tiles = self.tc.get_tiles(position, velocity)
        current_action, action_value = self.select_action(active_tiles)

        td_target = reward + self.discount * action_value
        td_error = td_target - self.w[self.last_action][self.previous_tiles].sum()
        self.w[self.last_action][self.previous_tiles] += self.step_size * td_error
        
        
#         active_tiles = self.tc.get_tiles(position, velocity) 
#         current_action, action_value = self.select_action(active_tiles)
        
#         previous_action, previous_action_value = self.select_action(self.previous_tiles)
        
#         for previous_tile in self.previous_tiles:
#             self.w[previous_action][previous_tile] = self.w[previous_action][previous_tile] + (self.step_size * (reward + (self.discount * action_value) - previous_action_value))
            
            
        ### END SOLUTION
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        self.steps += 1
        return self.last_action

    def agent_end(self, reward, staste, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Update self.w at self.previous_tiles and self.previous action
        # using the reward, self.discount, self.w,
        # self.step_size, and the Sarsa update from the textbook
        # Hint - there is no action_value used here because this is the end
        # of the episode.
        
        ### BEGIN SOLUTION
        if append_buffer:
            td_target = reward
            td_error = td_target - self.w[self.last_action][self.previous_tiles].sum()
            self.w[self.last_action][self.previous_tiles] += self.step_size * td_error
        ### END SOLUTION
        
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass