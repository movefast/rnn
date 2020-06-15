import numpy as np
import agent
import torch
import torch.nn as nn
import torch.nn.functional as F
# from replay_buffer_episodic import ReplayMemory, Transition
from buffer.replay_buffer import ReplayMemory, Transition
from buffer.prioritized_memory import Memory

criterion = torch.nn.SmoothL1Loss(reduction='none')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2h.weight.data.fill_(0)
        # self.i2h.bias.data.fill_(0)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.i2o.weight.data.fill_(0)
        # self.i2o.bias.data.fill_(0)
        self.tanh = nn.Tanh()
        self.actions = nn.Parameter(torch.normal(0, .01, (4, hidden_size)))
        # self.att = nn.Linear(hidden_size, 1)
        self.lambdas = nn.Parameter(torch.zeros(10))
        # self.att = nn.Linear(hidden_size, )
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp, hidden):
        output = []
        hiddens = []
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]
            combined = torch.cat((x, hidden), 1)
            hidden = self.i2h(combined)
            hidden = self.tanh(hidden)
            hiddens.append(hidden)
            # hidden = self.tanh(hidden)
            output.append(self.i2o(combined))
            # output = F.softmax(output)
            # q_vals = F.softmax(output[-1], -1)
            # q_idx = q_vals.max(1)[1]
            # hidden *= 1 + self.actions[q_idx]
        return output[-1], hiddens[-1]

    def batch(self, inp, hidden, discount_batch, action_batch):
        output = []
        hiddens = []
        if len(inp.size()) == 2:
            inp = inp.unsqueeze(1)
        for i in range(inp.size(0)):
            x = inp[i]
            combined = torch.cat((x, hidden), 1)
            hidden = self.i2h(combined)
            hidden = self.tanh(hidden)
            output.append(self.i2o(combined))
            # q_vals = F.softmax(output[-1], -1)
            # q_idx = q_vals.max(1)[1]
            # hidden *= 1 + self.actions[q_idx]
            hidden = hidden * (1 + self.actions[action_batch[i]])

            hiddens.append(hidden.detach())

            if discount_batch[i].item() == 0:
                hidden = self.initHidden()
        return torch.cat(output), hiddens

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)


class RNNAgent(agent.BaseAgent):
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

        # self.rnn = SimpleRNN(self.num_states+1, self.num_states+1,self.num_actions).to(device)
        # self.target_rnn = SimpleRNN(self.num_states+1, self.num_states+1,self.num_actions).to(device)
        self.rnn = SimpleRNN(self.num_states+1, self.num_states+1, self.num_actions).to(device)
        self.target_rnn = SimpleRNN(self.num_states+1, self.num_states+1, self.num_actions).to(device)
        # self.rnn = nn.RNN(self.num_states+1, self.num_states+1).to(device)
        # self.target_rnn = nn.RNN(self.num_states+1, self.num_states+1).to(device)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.step_size)
        self.buffer = Memory(1000)
        self.tau = .5
        self.flag = False
        self.train_steps = 0    

    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.Tensor(state).to(device)

        if self.is_door is None or is_door is True:
            self.is_door = int(is_door)
        else:
            self.is_door = self.is_door * .9 + is_door * .1
        # self.is_door = int(is_door)
        is_door = torch.Tensor([float(self.is_door)]).to(device)

        return torch.cat([state, is_door])[None, ...]

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
        self.hidden = self.rnn.initHidden()
        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        with torch.no_grad():
            self.hidden *= 1 + self.rnn.actions[action]

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

        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            target_q, _ = self.target_rnn(state, self.hidden)
        current_q.squeeze_()

        error = torch.abs(self.prev_action_value - reward - self.discount * target_q.max(1)[0]).item()
        self.buffer.add(error, self.prev_state, self.prev_action, reward, self.hidden.detach(), self.discount)

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        with torch.no_grad():
            self.hidden *= 1 + self.rnn.actions[action]

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        if len(self.buffer) > 20:# and self.steps % 5 == 0:# and self.epsilon == .1:
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
            error = torch.abs(self.prev_action_value - reward).item()
            self.buffer.add(error, self.prev_state, self.prev_action, reward, self.hidden.detach(), 0)
            self.flag = True

        if len(self.buffer) > 20:
            self.batch_train()

    def batch_train(self):
        self.train_steps += 1
        self.rnn.train()
        transitions, idxs, is_weight = self.buffer.sample_successive(11)

        # batch = transitions[0]        
        batch = Transition(*zip(*transitions))

        next_discount_batch = torch.FloatTensor(batch.discount[1:]).to(device)

        episode_bound = (next_discount_batch == 0).nonzero()
        # if self.flag:
        #     import pdb; pdb.set_trace()
        if episode_bound.nelement() > 0:
            boundry = episode_bound[0].item()
            
            if boundry <= 1:
                return
            batch = Transition(*zip(*transitions[:boundry]))
            next_discount_batch = torch.FloatTensor(batch.discount[1:]).to(device)

        discount_batch = torch.FloatTensor(batch.discount[:-1]).to(device)

        state_batch = torch.cat(batch.state[:-1])
        next_state_batch = torch.cat(batch.state[1:])

        action_batch = torch.LongTensor(batch.action[:-1]).view(-1, 1).to(device)
        next_action_batch = torch.LongTensor(batch.action[1:]).view(-1, 1).to(device)

        reward_batch = torch.FloatTensor(batch.reward[:-1]).to(device)
        # hidden_batch = torch.cat(batch.hidden)
        hidden_batch = batch.hidden[0]
        next_hidden_batch = batch.hidden[1] # or after rnn next_hidden_batch[0]

        h_batch = torch.cat(batch.hidden[1:])



        current_q, _ = self.rnn.batch(state_batch, hidden_batch, discount_batch, action_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            new_q, _ = self.target_rnn.batch(next_state_batch, next_hidden_batch, next_discount_batch, next_action_batch)
        max_q = new_q.max(1)[0]
        # max_q = new_q.gather(1, next_action_batch).squeeze_()    
        # target = reward_batch.clone()
        # target += discount_batch * max_q
        target = torch.zeros_like(max_q)

        for i in range(len(max_q)):
            discount_rates = torch.cumprod(discount_batch[i:], 0)

            reward_discount_rates = torch.zeros_like(discount_rates)
            reward_discount_rates[0] = 1
            reward_discount_rates[1:] = discount_rates[:-1]

            discounted_returns = torch.cumsum(reward_discount_rates * reward_batch[i:], 0)

            discounted_action_value_ests = discount_rates * max_q[i:]
            # 1)
            # target[i] = torch.mean(discounted_returns + discounted_action_value_ests)
            # 2)
            temp = discounted_returns + discounted_action_value_ests
            # lambdas = F.softmax(self.rnn.att(h_batch[i:]).view(-1))
            # 3)
            # lambdas = torch.ones(len(discounted_returns)).to(device)
            # lambdas[1:] = 0.9
            # lambdas = torch.cumprod(lambdas, 0)
            # target[i] = lambdas[-1] * temp[-1]
            # if len(lambdas) > 1:
            #     target[i] += torch.sum(lambdas[:-1] * temp[:-1]) * (1-0.9)
            # 4)
            lambdas = F.softmax(self.rnn.lambdas[i:len(max_q)]).view(-1)
            target[i] += torch.sum(lambdas * temp)


        target = target.view(-1, 1)

        temp = criterion(q_learning_action_values, target)
        loss = torch.Tensor(is_weight[:len(max_q)]).to(device) @ temp
        errors = torch.abs((q_learning_action_values - target).squeeze_(-1))
        for i in range(len(max_q)):
            self.buffer.update(idxs[i], errors[i].item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.rnn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.train_steps % 100 == 0:
            self.update()   

    def update(self):
        # target network update
        for target_param, param in zip(self.target_rnn.parameters(), self.rnn.parameters()):
            target_param.data.copy_(
                self.tau * param + (1 - self.tau) * target_param)

    # def update(self):
    #     self.target_rnn.load_state_dict(self.rnn.state_dict())
