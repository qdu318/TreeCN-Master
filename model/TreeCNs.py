import math
import torch
import torch.nn as nn
from model.TreeGradient import TreeGradient



class TreeCNs(nn.Module):
    def __init__(
            self,
            num_nodes,
            spatial_channels,
            timesteps_output,
            max_node_number
    ):
        super(TreeCNs, self).__init__()
        self.spatial_channels = spatial_channels
        self.Theta1 = nn.Parameter(torch.FloatTensor(1, spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.fully = nn.Linear(192, timesteps_output)
        self.TreeGradient = TreeGradient(num_nodes=num_nodes, max_node_number=max_node_number)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, NATree, X):
        tree_gradient = self.TreeGradient(NATree)
        lfs = torch.einsum("ij,jklm->kilm", [tree_gradient, X.permute(1, 0, 2, 3)])

        t2 = torch.tanh(torch.matmul(lfs, self.Theta1))
        # t2 = torch.matmul(lfs, self.Theta1)

        out3 = self.batch_norm(t2)
        # out3 = F.sigmoid(out3)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        return out4