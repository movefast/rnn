import numpy as np
import agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayMemory, Transition
from gwt import GridWorldTileCoder

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearAgent(agent.BaseAgent):
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
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.nn = nn.Linear(self.num_states,
                            self.num_actions, bias=False).to(device)
        self.target_nn = nn.Linear(
            self.num_states, self.num_actions, bias=False).to(device)
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(1000)
        self.tau = 0.01
        self.gwt = GridWorldTileCoder(num_tilings=self.num_states)

    def get_state_feature(self, state):
        # state, is_door = state
        # state = np.eye(self.num_states)[state]
        # state = torch.Tensor(state).to(device)
        # state = torch.Tensor([state]).to(device)
        # is_door = torch.Tensor([int(is_door)]).to(device)
        # if self.feature is None:
        #     self.feature = state
        # else:
        #     self.feature = self.feature * 0.9 + state * 0.1
        
        # return torch.Tensor(tiles(self.iht, self.num_states, self.feature))[None, ...].to(device)
        state, is_door = state
        if self.feature is None:
            self.feature = state, is_door
        else:
            # self.feature = self.feature[0] * 0.9 + state * 0.1, self.feature[1] * .9 + is_door * .1
            if is_door == True:
                self.feature = state, is_door
            else:
                self.feature = state, self.feature[1] * .9 + is_door * .1
        return torch.Tensor(self.gwt.get_tiles(*self.feature))[None, ...].to(device)
        # return torch.Tensor(tiles(self.iht, self.num_states, torch.cat([state, is_door])))[None, ...].to(device)
        # return torch.cat([state, is_door])[None, ...]

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
        self.feature = None
        state = self.get_state_feature(state)
        with torch.no_grad():
            current_q = F.softmax(self.nn(state))
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.epsilon = 1
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps = 0
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        # Choose action using epsilon greedy.
        state = self.get_state_feature(state)
        self.buffer.push(self.prev_state, self.prev_action, state, reward)

        with torch.no_grad():
            current_q = F.softmax(self.nn(state))
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.epsilon *= 0.98
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        if len(self.buffer) > 10:
            self.batch_train()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        if self.steps < 500 and append_buffer:
            self.buffer.push(self.prev_state, self.prev_action, state, reward)
#         if len(self.buffer) == 100:
        self.batch_train()

    def batch_train(self):
        self.nn.train()
        transitions = self.buffer.sample(10)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        non_final_mask = (reward_batch == -1)

        current_q = self.nn(torch.cat(batch.state))
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            new_q = self.nn(new_state_batch)
        max_q = new_q.max(1)[0]
        target = reward_batch
        target[non_final_mask] += self.discount * max_q[non_final_mask]
        target = target.view(-1, 1)
        loss = criterion(q_learning_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(
                self.tau * param + (1 - self.tau) * target_param)
