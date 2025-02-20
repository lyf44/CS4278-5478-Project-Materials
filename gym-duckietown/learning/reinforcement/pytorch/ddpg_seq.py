import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 5

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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
        flat_size = 64 * 7 * 10

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(15, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.lr(self.pool1(self.conv1(x)))
        x = self.lr(self.pool2(self.conv2(x)))
        x = self.lr(self.pool3(self.conv3(x)))
        # print(x.shape)
        # print(x.size(0))
        # x = x.view(x.size(0), -1)  # flatten
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # this is the vanilla implementation
        # but we're using a slightly different one
        # x = self.max_action * self.tanh(self.lin2(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
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

        flat_size = 64 * 7 * 10

        self.lr = nn.LeakyReLU()
        # self.relu = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(15, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = self.lr(self.pool1(self.conv1(states)))
        x = self.lr(self.pool2(self.conv2(x)))
        x = self.lr(self.pool3(self.conv3(x)))
        # x = self.bn4(self.lr(self.conv4(x)))
        # x = x.view(x.size(0), -1)  # flatten
        x = x.reshape(x.shape[0], -1)
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)

        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, net_type):
        super(DDPG, self).__init__()
        print("Starting DDPG init")
        assert net_type in ["cnn", "dense"]
        self.pred_buffer = []
        self.state_dim = state_dim

        if net_type == "dense":
            self.flat = True
            self.actor = ActorDense(state_dim, action_dim, max_action).to(device)
            self.actor_target = ActorDense(state_dim, action_dim, max_action).to(device)
        else:
            self.flat = False
            self.actor = ActorCNN(action_dim, max_action).to(device)
            self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        print("Initialized Target+Opt [Actor]")
        if net_type == "dense":
            self.critic = CriticDense(state_dim, action_dim).to(device)
            self.critic_target = CriticDense(state_dim, action_dim).to(device)
        else:
            self.critic = CriticCNN(action_dim).to(device)
            self.critic_target = CriticCNN(action_dim).to(device)
        print("Initialized Critic")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        print("Initialized Target+Opt [Critic]")

    def predict(self, state):
        # state = state[0]
        # print(np.shape(state))
        if np.size(self.pred_buffer) == 0:
            seq = [state for _ in range(SEQUENCE_LENGTH)]
            states = np.concatenate(seq)
        else:
            states = np.concatenate((self.pred_buffer[3:], state))
        self.pred_buffer = states
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        # print(states.shape)
        assert states.shape[0] == 15

        if self.flat:
            states = torch.FloatTensor(states.reshape(1, -1)).to(device)
        else:
            states = torch.FloatTensor(np.expand_dims(states, axis=0)).to(device)

        states = states.detach()
        return self.actor(states).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            assert state.shape == (batch_size, 15, 80, 60) and next_state.shape == (batch_size, 15, 80, 60)

            # Compute the target Q value
            # print(next_state.shape)
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

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))
        print("Saved Critic")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))
