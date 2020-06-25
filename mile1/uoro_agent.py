import torch
from torch.autograd import grad, Variable
import numpy as np
import agent
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Param2Vec():	
    def __init__(self, model):	
        """	
        get a list of trainable variables	
        """	
        self.param_list = []	
        # self.state_param_list = []	
        self.size_list = []	
        for p in model.parameters():	
            if p.requires_grad:	
                self.param_list.append(p)	
                self.size_list.append(p.size())	
        # for p in model.rnn_layer.parameters():	
        #     if p.requires_grad:	
        #         self.state_param_list.append(p)	
        self.num_list = len(self.param_list)	
    def merge(self, var_list):	
        """	
        merge a list of variables to a vector	
        """	
        assert len(var_list) == len(self.size_list)	
        theta_list = []	
        for i in range(len(var_list)):	
            var = var_list[i]	
            if var is not None:	
                theta_list.append(var.flatten())	
            else:	
                theta_list.append(torch.zeros(self.size_list[i]).flatten().to(device))	
        return torch.cat(theta_list)	
    def split(self, var_vec):	
        """	
        split a vec to a list	
        """	
        var_list = []	
        count = 0	
        for i in range(len(self.size_list)):	
            prod_size = np.prod(self.size_list[i])	
            var_list.append(var_vec[count:(count+prod_size)].reshape(self.size_list[i]))	
            count += prod_size	
        return var_list


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        # self.actions = nn.Parameter(torch.normal(0, .01, (4, hidden_size)))

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
            output.append(self.i2o(combined))
        return output[-1], hiddens[-1]

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(device)
        # TODO: legacy; not sure if we need second dim here for uoro gradient as well
        # return torch.zeros(1, 1, self.hidden_size).to(device)

class UOROAgent(agent.BaseAgent):
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.optimizer_type = 'RMSprop'

        epsilon_perturbation=1e-7
        epsilon_stability=1e-7

        assert self.optimizer_type in ['SGD', 'RMSprop']
        self.hidden_size = self.num_states+1
        self.device = device
        self.rnn = SimpleRNN(self.num_states+1, self.num_states+1, self.num_actions).to(device)
        self.epsilon_perturbation = epsilon_perturbation
        self.epsilon_stability = epsilon_stability
        self.criterion = torch.nn.SmoothL1Loss()
        if self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.step_size)
        else:
            self.optimizer = torch.optim.RMSprop(self.rnn.parameters(), lr=self.step_size, alpha=0.99)
        # lambda1 = lambda epoch: 1 / (1 + 0.003 * np.sqrt(epoch))
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.99)
        self.nn_param = Param2Vec(self.rnn)
        self.s_tilda = None
        self.theta_tilda = None
        # self._state = None

    # def initialize_state(self):
    #     self._state = torch.zeros(size=[1, 1, self.hidden_size], requires_grad=True).float()

    # def predict(self, x):
    #     state_old = self._state.clone().detach()
    #     y_pred, state_new = self.rnn(x, state_old)
    #     self._state = state_new
    #     return y_pred, state_old, state_new

    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.Tensor(state).to(device)

        # if self.is_door is None or is_door is True:
        #     self.is_door = int(is_door)
        # else:
        #     self.is_door = self.is_door * .9 + is_door * .1
        self.is_door = int(is_door)
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
            current_q, new_hidden = self.rnn(state, self.hidden)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]
        self.prev_hidden = self.hidden
        self.hidden = new_hidden
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

        # self.buffer.push(self.prev_state, self.prev_action, reward, self.hidden.detach(), self.discount)
        with torch.no_grad():
            current_q, new_hidden = self.rnn(state, self.hidden)
        current_q.squeeze_()
        self.train(self.prev_state, reward + self.discount * current_q.max(), self.prev_hidden, self.prev_action)

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        # with torch.no_grad():
        #     self.hidden *= 1 + self.rnn.actions[action]
        
        self.prev_hidden = self.hidden
        self.hidden = new_hidden
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        self.train(self.prev_state, torch.tensor(reward).to(device), self.prev_hidden, self.prev_action)
        # if append_buffer:
        #     self.buffer.push(self.prev_state, self.prev_action, reward, self.hidden.detach(), 0)
        #     self.flag = True

    def train(self, x, y, state, action):
        """
        x: input of size (1, 1, input_size)
        s: previous recurrent state of size (1, 1, hidden_size)
        y: target of size (1,)
        self.s_toupee: column vector of size (state, )
        self.theta_toupee: row vector of size (params, )
        """
        # print('---')
        state_old = state.clone().detach().requires_grad_(True)
        y_1, state_new = self.rnn(x, state_old)
        # print(x.size())
        # print(y.size())
        # print(y_1[-1].size())
        loss = self.criterion(y_1.squeeze()[action], y)
        # update weights
        delta_s = grad(loss, state_old, retain_graph=True)[0]
        # TODO allow_unused=True; might be because action select 
        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True, allow_unused=True)
        delta_theta_vec = self.nn_param.merge(delta_theta)
        if self.s_tilda is None or self.theta_tilda is None:
            self.s_tilda = Variable(torch.zeros(*state_old.squeeze().size())).to(device)  # (batch_size, state_dim)
            self.theta_tilda = Variable(torch.zeros(*delta_theta_vec.size())).to(device)  # (n_params, )
        # print(self.s_tilda.size())
        # print(self.theta_tilda.size())
        g_t1 = torch.dot(delta_s.squeeze(), self.s_tilda) * self.theta_tilda + delta_theta_vec
        g_t1_list = self.nn_param.split(g_t1)
        # ForwardDiff
        state_old_perturbed = state_old + self.s_tilda * self.epsilon_perturbation  #.detach()
        state_new_perturbed = self.rnn(x, state_old_perturbed)[1]
        s_forwarddiff = (state_new_perturbed - state_new)/self.epsilon_perturbation
        # Backprop
        nu_vec = Variable(torch.round(torch.rand(*state_old.size())) * 2 - 1).to(device)
        delta_theta_g = grad(outputs=state_new, inputs=self.nn_param.param_list, grad_outputs=nu_vec, allow_unused=True,
                             retain_graph=True)
        delta_theta_g_vec = self.nn_param.merge(delta_theta_g)
        rho_0 = torch.sqrt(self.theta_tilda.norm()/(s_forwarddiff.norm() + self.epsilon_stability)) + self.epsilon_stability
        rho_1 = torch.sqrt(delta_theta_g_vec.norm()/(nu_vec.norm() + self.epsilon_stability)) + self.epsilon_stability
        self.s_tilda = (rho_0 * s_forwarddiff.squeeze() + rho_1 * nu_vec.squeeze()).detach()
        self.theta_tilda = (self.theta_tilda / rho_0 + delta_theta_g_vec / rho_1).detach()
        # self.optimizer.zero_grad()
        for i in range(len(self.nn_param.param_list)):
            self.nn_param.param_list[i].grad = g_t1_list[i]
        self.optimizer.step()
        # self.scheduler.step()
        # self._state = state_new
        return loss.item(), state_old, state_new
        # return y_1