import numpy as np
import agent
import torch
import torch.nn as nn
import torch.nn.functional as F
# from replay_buffer_episodic import ReplayMemory, Transition
from replay_buffer import ReplayMemory, Transition

criterion = torch.nn.MSELoss()
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
            q_vals = F.softmax(output[-1], -1)
            q_idx = q_vals.max(1)[1]
            hidden *= 1 + self.actions[q_idx]
        if len(hiddens) == 1:
            return torch.cat(output), hiddens[0] 
        else:
            return torch.cat(output), hiddens[0]

    def batch(self, inp, hidden, non_final_mask):
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
            # output = self.softmax(output)
            # q_vals = F.softmax(output[-1], -1)
            # q_idx = q_vals.max(1)[1]
            # hidden *= 1 + self.actions[q_idx]
            if non_final_mask[i].item() is False:
                hidden = self.initHidden()
        if len(hiddens) == 1:
            return torch.cat(output), hiddens[0] 
        else:
            return torch.cat(output), hiddens[0]

    def batch_action(self, inp, hidden, non_final_mask, action_batch):
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
            # output = self.softmax(output)
            q_vals = F.softmax(output[-1], -1)
            q_idx = action_batch[i]
            hidden *= 1 + self.actions[q_idx]
            if non_final_mask[i].item() is False:
                hidden = self.initHidden()
        if len(hiddens) == 1:
            return torch.cat(output), hiddens[0] 
        else:
            return torch.cat(output), hiddens[0]

    # def forward(self, input, hidden):
    #     combined = torch.cat((input, hidden), 1)
    #     hidden = self.tanh(self.i2h(combined))
    #     output = self.i2o(combined)
    #     return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(hidden_size, output_size)
#         self.h2h = nn.Linear(hidden_size, hidden_size)
#         self.tanh = nn.Tanh()
#         self.actions = nn.Parameters(4,hidden_size)

#     # def forward(self, input, hidden):
#     #     combined = torch.cat((input, hidden), 1)
#     #     hidden = self.tanh(self.i2h(combined))
#     #     output = self.i2o(hidden)
#     #     hidden = self.tanh(self.h2h(hidden))
#     #     return output, hidden

#     def forward(self, inp, hidden):
#         output = []
#         if len(inp.size()) == 2:
#             inp = inp.unsqueeze(1)
#         for i in range(inp.size(0)):
#             x = inp[i]
#             combined = torch.cat((x, hidden), 1)
#             hidden = self.tanh(self.i2h(combined))
#             output.append(self.i2o(hidden))
#             hidden = self.h2h(hidden)
#         return torch.cat(output), self.tanh(hidden)

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size).to(device)


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
        # self.target_rnn = nn.RNN(
        #     self.num_states+1, self.num_states+1).to(device)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(500)
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
        state = self.get_state_feature(state)
        self.hidden = self.rnn.initHidden()
        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()
        # self.epsilon = 1
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        # self.states = []
        # self.actions = []
        # self.new_states = []
        # self.rewards = []
        # self.hiddens = []
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
        # temp = state
        state = self.get_state_feature(state)
        # self.states.append(self.prev_state)
        # self.actions.append(self.prev_action)
        # self.new_states.append(state)
        # self.rewards.append(reward)
        # self.hiddens.append(self.hidden.detach())
        self.buffer.push(self.prev_state, self.prev_action, state, reward, self.hidden.detach())

        with torch.no_grad():
            current_q, self.hidden = self.rnn(state, self.hidden)
            # if torch.isnan(self.hidden).any():
            #     print(temp)
            #     print(self.hidden)
            current_q = F.softmax(current_q, -1)
        current_q.squeeze_()
        # self.epsilon = max(0.1, self.epsilon * 0.98)

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
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
        # self.states.append(self.prev_state)
        # self.actions.append(self.prev_action)
        # self.new_states.append(torch.Tensor([np.zeros(self.num_states+1)]).to(device))
        # self.rewards.append(reward)
        # self.hiddens.append(self.hidden.detach())
        # if append_buffer:
        #     self.buffer.push(self.states, self.actions, self.new_states, self.rewards, self.hiddens)
        if append_buffer:
            # self.buffer.push(self.prev_state, self.prev_action, torch.Tensor(
            #     [np.zeros(self.num_states+1)]).to(device), reward, self.hidden.detach())
            self.buffer.push(self.prev_state, self.prev_action, state, reward, self.hidden.detach())
            self.flag = True

        if len(self.buffer) > 20:
            self.batch_train()

    def batch_train(self):
        self.train_steps += 1
        self.rnn.train()
        transitions = self.buffer.sample_successive(20)
        # import pdb; pdb.set_trace()
        # batch = transitions[0]        
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        # import pdb
        # pdb.set_trace()
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        # hidden_batch = torch.cat(batch.hidden)
        hidden_batch = batch.hidden[0]
        non_final_mask = (reward_batch == -1)
        current_q, _ = self.rnn.batch(state_batch[0:1], hidden_batch, non_final_mask[0:1])
        q_learning_action_values = current_q.gather(1, action_batch[0:1])
        with torch.no_grad():
            new_q, _ = self.target_rnn.batch(new_state_batch[-2:-1], batch.hidden[-1], non_final_mask[-2:-1])
        # max_q = new_q.max(1)[0]
        max_q = new_q.gather(1, action_batch[-2:-1]).squeeze_()    
        target = reward_batch
        # target[non_final_mask] += self.discount * max_q[non_final_mask]
        discount_rates = np.ones(20)
        discount_rates[1:] = self.discount
        discount_rates = torch.from_numpy(np.cumprod(discount_rates)).float().to(device)
        target = torch.sum(discount_rates * target)
        if non_final_mask[-1].item():
            target += discount_rates[-1] * self.discount * max_q#[-1]

        target = target.view(-1, 1)
        loss = criterion(q_learning_action_values[0], target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.rnn.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.train_steps % 100 == 0:
            self.update()   

#     def batch_train(self):
#         self.rnn.train()
#         seq_len = 30
#         transitions = self.buffer.sample_successive(seq_len)
#         loss = 0
#         batch = Transition(*zip(*transitions))
#         target = 0
#         non_final_mask = torch.ones(1)
#         for i in range(seq_len):

#             state_batch =  batch.state[i]
#             action_batch = torch.LongTensor([batch.action[i]]).to(device)
#             new_state_batch = batch.new_state[i]
#             reward_batch = torch.FloatTensor([batch.reward[i]]).to(device)
# #             if i == 0:
#                 # hidden_batch = torch.zeros_like(batch.hidden[0])
#             hidden_batch = batch.hidden[i]
# #             import pdb; pdb.set_trace()
#             non_final_mask = non_final_mask and (reward_batch == -1)

#             current_q, hidden_batch = self.rnn(state_batch, hidden_batch)
#             current_q = current_q.squeeze()
#             # set terminal hidden state to 0 in training
#             hidden_batch.detach_()
#             hidden_batch[~non_final_mask] = 0

#             q_learning_action_values = current_q.gather(0,action_batch).view(-1,1)
#             target += self.discount ** (i+1) * reward_batch

#         with torch.no_grad():
#             new_q, _ = self.target_rnn(new_state_batch, hidden_batch)
#         max_q = new_q.max(1)[0]
#         target[non_final_mask] += self.discount ** (seq_len) * max_q[non_final_mask]
#         target = target.view(-1,1)
#         # print(loss)
# #             print(q_learning_action_values, target, max_q, end='\r')
#         # print(max_q)

#         loss = criterion(q_learning_action_values, target)
#         print(loss,end='\r')
#         self.optimizer.zero_grad()
#         loss.backward()
#         for param in self.rnn.parameters():
#             if param.grad is not None:
#                 param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()
#         self.update()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_rnn.parameters(), self.rnn.parameters()):
            target_param.data.copy_(
                self.tau * param + (1 - self.tau) * target_param)

    # def update(self):
    #     self.target_rnn.load_state_dict(self.rnn.state_dict())
