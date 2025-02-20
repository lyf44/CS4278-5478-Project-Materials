import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

class ActorDense(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * self.tanh(self.l3(x))
        return x


class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 6 * 6

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 128)
        self.lin2 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, x):
        # print(x.max())
        x = self.bn1(self.lr(self.conv1(x)))
        # print(x.max())
        x = self.bn2(self.lr(self.conv2(x)))
        # print(x.max())
        x = self.bn3(self.lr(self.conv3(x)))
        # print(x.max())
        x = self.bn4(self.lr(self.conv4(x)))
        # print(x.max())
        try:
            x = x.view(x.size(0), -1)  # flatten
        except RuntimeError:
            x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        # print(x.max())
        x = self.bn5(self.lr(self.lin1(x)))
        # print(x.max())

        # because we don't want our duckie to go backwards
        x = self.lin2(x)

        # If we want the duckie to go backwards, change to two tanh instead of one sigm and one tanh
        # print(x)
        # x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 0] = (self.max_action * self.tanh(x[:, 0]) + 1) / 2  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1])

        return x


class CriticDense(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticDense, self).__init__()

        state_dim = functools.reduce(operator.mul, state_dim, 1)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x


class CriticCNN(nn.Module):
    def __init__(self, action_dim):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 6 * 6

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 2, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        try:
            x = x.view(x.size(0), -1)  # flatten
        except RuntimeError:
            x = x.reshape(x.size(0), -1)
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)

        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, net_type):
        super(DDPG, self).__init__()
        assert net_type in ["cnn", "dense"]

        self.state_dim = state_dim

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        else:
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action).to(device)
            self.actor_target = ActorCNN(action_dim, max_action).to(device)

        self.actor.apply(init_weights)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        if net_type == "dense":
            self.critic = CriticDense(state_dim, action_dim).to(device)
            self.critic_target = CriticDense(state_dim, action_dim).to(device)
        else:
            self.critic = CriticCNN(action_dim).to(device)
            self.critic_target = CriticCNN(action_dim).to(device)

        self.critic.apply(init_weights)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def set_mode(self, training=True):
        if training:
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()
        else:
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic_target.eval()

    def predict(self, state):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)

        state = state.detach()
        action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            print("actor_loss: {}, critic_loss : {}".format(actor_loss, critic_loss))

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))