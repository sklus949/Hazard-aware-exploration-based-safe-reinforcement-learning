import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
GAMMA = 0.99


class ValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, device):
        super(ValueNet, self).__init__()
        self.device = device
        self.self_state_dim = self_state_dim
        self.action_dim = action_dim
        self.global_state_dim = hidden_dim
        self.mlp1 = nn.Linear(state_dim, hidden_dim).to(self.device)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.attention = nn.Linear(hidden_size, 1).to(self.device)
        input_dim = hidden_dim + self.self_state_dim + action_dim
        self.linear2 = nn.Linear(input_dim, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.network1 = nn.Linear(hidden_size, 1).to(self.device)
        self.network2 = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, state, action):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = torch.relu(self.mlp1(state.view((-1, size[2]))))
        mlp2_output = self.mlp2(mlp1_output)
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
            contiguous().view(-1, self.global_state_dim)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)

        features = mlp2_output.view(size[0], size[1], -1)

        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        action = action.reshape(-1, self.action_dim)
        x = torch.cat((joint_state, action), -1)
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        q1 = self.network1(x)
        q2 = self.network2(x)
        return q1, q2


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, device):
        super(PolicyNet, self).__init__()
        self.device = device
        self.self_state_dim = self_state_dim
        self.action_dim = action_dim
        self.global_state_dim = hidden_dim
        self.mlp1 = nn.Linear(state_dim, hidden_dim).to(self.device)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.attention = nn.Linear(hidden_size, 1).to(self.device)
        input_dim = hidden_dim + self.self_state_dim
        self.linear2 = nn.Linear(input_dim, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.mean_linear = nn.Linear(hidden_size, action_dim).to(self.device)
        self.log_std_linear = nn.Linear(hidden_size, action_dim).to(self.device)

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = torch.relu(self.mlp1(state.view((-1, size[2]))))
        mlp2_output = self.mlp2(mlp1_output)
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
            contiguous().view(-1, self.global_state_dim)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)

        features = mlp2_output.view(size[0], size[1], -1)

        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        joint_state = torch.cat([self_state, weighted_feature], dim=1)

        x = torch.relu(self.linear2(joint_state))
        x = torch.relu(self.linear3(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state, phase='train'):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        if phase == 'train':
            action = dist.rsample()
        else:
            action = dist.mean
        action_ = torch.tanh(action)

        log_prob = dist.log_prob(action) - torch.log(1 - action_.pow(2) + 1e-6)

        return action_, log_prob.sum(1, keepdim=True)


class HESRL:
    def __init__(self, state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, critic_lr, actor_lr, alpha_lr,
                 device):
        super().__init__()
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.device = device

        self.critic = ValueNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)
        self.target_critic = ValueNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor = PolicyNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)

        self.critic_vr = ValueNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)
        self.target_critic_vr = ValueNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)
        self.target_critic_vr.load_state_dict(self.critic_vr.state_dict())
        self.actor_vr = PolicyNet(state_dim, action_dim, self_state_dim, hidden_dim, hidden_size, self.device)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.log_alpha_vr = torch.zeros(1, requires_grad=True, device=self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.critic_optimizer_vr = torch.optim.Adam(self.critic_vr.parameters(), lr=critic_lr)
        self.actor_optimizer_vr = torch.optim.Adam(self.actor_vr.parameters(), lr=actor_lr)
        self.optimizer_vr = torch.optim.Adam([self.log_alpha_vr], lr=alpha_lr)

        self.memory1 = Memory(state_shape=[5, 13], action_shape=[1, 2], capacity=100000, device=self.device)
        self.memory2 = Memory(state_shape=[5, 13], action_shape=[1, 2], capacity=100000, device=self.device)
        self.memory_vr = Memory(state_shape=[5, 13], action_shape=[1, 2], capacity=200000, device=self.device)

        self.batch_size1 = 64
        self.batch_size2 = 64
        self.batch_size = 128
        self.gamma = GAMMA
        self.tau = 0.005

    def predict(self, state, phase):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            action, _ = self.actor.sample(state, phase)

        return action

    def predict_vr(self, state, phase):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            action, _ = self.actor_vr.sample(state, phase)

        return action

    @staticmethod
    def get_human_array(state):
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state])
                                  for human_state in state.human_states], dim=0)
        array = state_tensor[:, 9: 13]
        human_x = array[:, 0]
        human_y = array[:, 1]
        human_vx = array[:, 2]
        human_vy = array[:, 3]
        return human_x, human_y, human_vx, human_vy

    @staticmethod
    def get_robot_state(state):
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state])
                                  for human_state in state.human_states], dim=0)
        array = state_tensor[0, 0: 9]
        robot_x = array[0]
        robot_y = array[1]
        return robot_x, robot_y

    def transform(self, state):

        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state])
                                  for human_state in state.human_states], dim=0)
        state_tensor = self.rotate(state_tensor)
        return state_tensor

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

    def optim(self):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0

        state1, action1, reward1, next_state1, done1 = self.memory1.sample(self.batch_size1)
        state2, action2, reward2, next_state2, done2 = self.memory2.sample(self.batch_size2)
        state = torch.cat((state1, state2), dim=0)
        action = torch.cat((action1, action2), dim=0)
        reward = torch.cat((reward1, reward2), dim=0)
        next_state = torch.cat((next_state1, next_state2), dim=0)
        done = torch.cat((done1, done2), dim=0)

        with torch.no_grad():
            next_action, log_prob = self.actor.sample(next_state)
            q_t1, q_t2 = self.target_critic(next_state, next_action)
            q_target = torch.min(q_t1, q_t2)
            value_target = reward + (1 - done) * self.gamma * (q_target - self.alpha * log_prob)
        q_1, q_2 = self.critic(state, action)

        q1_loss_step = F.mse_loss(q_1, value_target)
        q2_loss_step = F.mse_loss(q_2, value_target)
        q_loss_step = q1_loss_step + q2_loss_step

        action, log_prob = self.actor.sample(state)

        q_b1, q_b2 = self.critic(state, action)

        qval_batch = torch.min(q_b1, q_b2)
        pi_loss_step = (self.alpha.detach() * log_prob - qval_batch).mean()
        alpha_loss_step = -self.alpha * (log_prob.detach() + self.target_entropy).mean()

        self.actor_optimizer.zero_grad()
        pi_loss_step.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        q_loss_step.backward()
        self.critic_optimizer.step()

        self.optimizer.zero_grad()
        alpha_loss_step.backward()
        self.optimizer.step()

        q1_loss += q1_loss_step.detach().item()
        q2_loss += q2_loss_step.detach().item()
        pi_loss += pi_loss_step.detach().item()
        alpha_loss += alpha_loss_step.detach().item()
        self.soft_update()

    def optim_vr(self):
        q1_loss, q2_loss, pi_loss, alpha_loss = 0, 0, 0, 0

        state, action, reward, next_state, done = self.memory_vr.sample(self.batch_size)
        with torch.no_grad():
            next_action, log_prob = self.actor_vr.sample(next_state)
            q_t1, q_t2 = self.target_critic_vr(next_state, next_action)
            q_target = torch.min(q_t1, q_t2)
            value_target = reward + (1 - done) * self.gamma * (q_target - self.alpha_vr * log_prob)
        q_1, q_2 = self.critic_vr(state, action)

        q1_loss_step = F.mse_loss(q_1, value_target)
        q2_loss_step = F.mse_loss(q_2, value_target)
        q_loss_step = q1_loss_step + q2_loss_step

        action, log_prob = self.actor_vr.sample(state)

        q_b1, q_b2 = self.critic_vr(state, action)

        qval_batch = torch.min(q_b1, q_b2)
        pi_loss_step = (self.alpha_vr.detach() * log_prob - qval_batch).mean()
        alpha_loss_step = -self.alpha_vr * (log_prob.detach() + self.target_entropy).mean()

        self.actor_optimizer_vr.zero_grad()
        pi_loss_step.backward()
        self.actor_optimizer_vr.step()

        self.critic_optimizer_vr.zero_grad()
        q_loss_step.backward()
        self.critic_optimizer_vr.step()

        self.optimizer_vr.zero_grad()
        alpha_loss_step.backward()
        self.optimizer_vr.step()

        q1_loss += q1_loss_step.detach().item()
        q2_loss += q2_loss_step.detach().item()
        pi_loss += pi_loss_step.detach().item()
        alpha_loss += alpha_loss_step.detach().item()
        self.soft_update_vr()

    def soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def soft_update_vr(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_critic_vr.parameters(), self.critic_vr.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def alpha_vr(self):
        return self.log_alpha_vr.exp()


class Memory(object):
    """Buffer to store environment transitions."""

    def __init__(self, state_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.counter = 0
        self.device = device
        self.state = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_state = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        index = self.counter % self.capacity
        np.copyto(self.state[index], state)
        np.copyto(self.actions[index], action)
        np.copyto(self.rewards[index], reward)
        np.copyto(self.next_state[index], next_state)
        np.copyto(self.done[index], done)
        self.counter += 1

    def sample(self, batch_size):
        index = np.random.choice(min(self.capacity, self.counter), batch_size)
        state = torch.as_tensor(self.state[index]).to(self.device)
        action = torch.as_tensor(self.actions[index]).to(self.device)
        reward = torch.as_tensor(self.rewards[index]).to(self.device)
        next_state = torch.as_tensor(self.next_state[index]).to(self.device)
        done = torch.as_tensor(self.done[index]).to(self.device)
        return state, action.squeeze(1), reward, next_state, done
