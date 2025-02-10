import torch
import torch.nn as nn
from libs.modeling.models import register_meta_arch, make_backbone, make_neck, make_generator
import torch.nn.functional as F
import math
import torch.nn.init as init

class ExpandY(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpandY, self).__init__()
        self.expand = nn.Linear(input_dim, output_dim)

    def forward(self, y):
        # y: (T, 2)
        # print("Input y:", y)  # Debug: print input
        y_expand = self.expand(y)  # pass input through the sequential model
        # print("Expanded y:", y_expand)  # Debug: print output
        return y_expand

@register_meta_arch('model_meta')
class MetaNet(nn.Module):
    def __init__(self, hx_dim, expand_dim, h_dim):
        super().__init__()

        self.num_classes = 2
        self.hdim = h_dim

        in_dim = hx_dim + expand_dim
        out_dim = 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, out_dim)
        )

        self.head = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )

        self.expand_y = ExpandY(input_dim=2, output_dim=expand_dim)
        #self.expand_loss = self.expand_loss = ExpandLoss(input_dim=1, output_dim=expand_dim)

    def forward(self, pseudo_reg_loss_vector, masked_hx_cat, y):

        # Create embeddings for the noisy labels
        y_expand = self.expand_y(y)

        # Concatenate the features and embeddings
        hin = torch.cat((masked_hx_cat, y_expand), dim=-1)
        direction = self.net(hin)
        direction_loss_cat = torch.cat((direction, pseudo_reg_loss_vector), dim=-1)

        # Pass through the network
        offsets = self.head(direction_loss_cat)
        out = y + offsets

        return out, offsets

