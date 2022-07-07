# encoding utf-8
import math
import torch
import torch.nn as nn


class TreeGradient(nn.Module):
    def __init__(self, num_nodes, max_node_number):
        super(TreeGradient, self).__init__()
        self.x = nn.Parameter(torch.FloatTensor(num_nodes, max_node_number))
        self.Conv = nn.Conv1d(in_channels=max_node_number, out_channels=max_node_number, kernel_size=2*1)
        self.Conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=max_node_number-num_nodes+1)
        self.fully = nn.Linear(max_node_number, num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.x.shape[1])
        self.x.data.uniform_(-stdv, stdv)

    def forward(self, NATree):
        total_x = None
        for i in range(NATree.shape[0]):
            first_single_x = torch.unsqueeze(torch.unsqueeze(NATree[i, -1], dim=0), dim=0)
            for j in range(NATree.shape[1] - 2, -1, -1):
                second_single_x = torch.unsqueeze(torch.unsqueeze(NATree[i, j], dim=0), dim=0)
                doueble_x = torch.cat((second_single_x, first_single_x), dim=1)
                first_single_x = self.Conv(doueble_x.permute(0, 2, 1)).permute(0, 2, 1)
                first_single_x = torch.tanh(first_single_x)
                first_single_x = first_single_x + second_single_x

            if i == 0:
                total_x = first_single_x
            else:
                total_x = torch.cat((total_x, first_single_x), dim=0)

        total_x = torch.tanh(total_x)
        result = torch.add(torch.squeeze(total_x, dim=1), self.x)
        result = torch.tanh(result)
        result = torch.unsqueeze(result, dim=0)
        result = self.Conv2(result.permute(1, 0, 2))
        result = torch.squeeze(result, dim=1)

        return result
