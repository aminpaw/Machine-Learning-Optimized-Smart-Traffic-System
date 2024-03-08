import torch
import time
import torch.nn as nn
import random

class DDPG(nn.Module):
    def __init__(self, obs_dim):
        super(DDPG, self).__init__()
        self.linear1 = nn.Linear(obs_dim, 256)  # Input layer
        self.linear2 = nn.Linear(256, 256)
        self.obs_relu = nn.ReLU()
        self.linear3 = nn.Linear(256, 3)
        self.obs_tanh = nn.Tanh()
        self.state= random.choice([0,1,2])

    def forward(self, x):
        x = self.linear1(x)
        x = self.obs_relu(x)
        x = self.linear2(x)
        x = self.obs_relu(x)
        x = self.linear3(x)
        x = self.obs_tanh(x)
        return x

    def load_weights(self, path):
        """
        Loads pre-trained weights from a file.

        Args:
            path (str): Path to the file containing the weights.
        """
        weights = torch.load(path)
        self.load_state_dict(weights)

    def get_action(self,x):
        x = torch.cat((x,torch.tensor([[self.state]]).float()),1)
        x = self.forward(x).detach().cpu().numpy()
        x = x[0].tolist()
        output = x.index(max(x))
        self.state=output

        return output


# Example usage:
model = DDPG(10)
model.load_weights("./actor.pkl")

# Example usage:
for i in range(10):
    x = torch.randn(1,9)
    print(model.get_action(x))
    time.sleep(1)


# print(torch.load("./actor.pkl"))



