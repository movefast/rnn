import numpy as np
import agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer_old import ReplayMemory, Transition

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.tanh = nn.ReLU()
        self.i2h = nn.Linear(input_size, input_size//2, bias=False)
        self.h2o = nn.Linear(input_size//2, output_size, bias=False)

    def forward(self, x):
        x = self.i2h(x)
        x = self.tanh(x)
        x = self.h2o(x)
        return x


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
        self.T = agent_init_info.get("T",10)
        # 1) exponential average
        # self.exp_decay = 2 / (1 + self.T)
        # 2) rl exponential decay
        self.exp_decay = 1 - 1 / self.T


        self.nn = SimpleNN(self.num_states+1, self.num_actions).to(device)
        # self.weights_init(self.nn)
        self.target_nn = SimpleNN(self.num_states+1, self.num_actions).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(1000)
        self.updates = 0

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.Tensor(state).to(device)
        self.is_door = int(is_door)
        is_door = torch.Tensor([float(self.is_door)]).to(device)
        if self.feature is None:
            self.feature = torch.cat([state, is_door])[None, ...]
        else:
            self.feature = self.feature + self.exp_decay * (torch.cat([state, is_door])[None, ...] - self.feature)
        return self.feature

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
        self.is_door = None
        self.feature = None
        state = self.get_state_feature(state)
        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
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
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1
        if len(self.buffer) > 1:
            self.batch_train()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        if append_buffer:
            self.buffer.push(self.prev_state, self.prev_action, state, reward)
        self.batch_train()

    def batch_train(self):
        self.updates += 1
        self.nn.train()
        transitions = self.buffer.sample(1)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        non_final_mask = (reward_batch == 0)

        current_q = self.nn(state_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            new_q = self.target_nn(new_state_batch)
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
        if self.updates % 100 == 0:
            self.update()

    # def update(self):
    #     # target network update
    #     for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
    #         target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())

    def update(self):
        self.target_nn.load_state_dict(self.nn.state_dict())