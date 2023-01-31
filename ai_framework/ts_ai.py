import os

import torch
from torch import nn
from torch import optim

from ai_framework.models.models import InceptionNet3


###
#
# Taken from:
# Johannes Alecke. Analyse und Optimierung von Angriffen auf tiefe neuronale Netze, Hochschule Bonn-Rhein-Sieg, 2020
#
###

class TSAI:

    def __init__(self,
                 name_for_save: str = None,
                 net: nn.Module = InceptionNet3(),
                 criterion=nn.CrossEntropyLoss(),
                 optimizer: optim = optim.Adam,
                 lr=1e-4):
        self.name = name_for_save

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = net

        self.net = self.net.to(self.device)
        self.criterion = criterion

        self.optimizer = optimizer(net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def load(self):
        print("=> loading checkpoint '{}'".format(self.name))
        dir_name = os.path.dirname(__file__)
        checkpoint = torch.load(f"{dir_name}/trained_models/{self.name}.pt")
        if type(checkpoint) is dict:
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            return start_epoch
        else:
            dir_name = os.path.dirname(__file__)
            state_dict = torch.load(f"{dir_name}/trained_models/{self.name}.pt")
            self.net.load_state_dict(state_dict=state_dict)

        self.net.to(self.device)

        print("=> Finished Loading\n")
