
import torch
import torch.nn as nn
import numpy as np

# SimLCR

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        """
        Calculate the loss of a batch. 
        The previous batch was passed in, 
        and the full scale and half scale in each batch are positive examples of each other
        """
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        """
        z_i:full scale sync patch, output (b*num_patches, projector_dim)
        z_j:half scale sync patch, output (b*num_patches, projector_dim)
        """
        # z_i.shape[0] the total number of features
        N = 2 * z_i.shape[0]    # double is to concat full-scale and half-scale
        z = torch.cat((z_i, z_j), dim=0)

        # calculate similarity and divide by temperature parameter
        z = nn.functional.normalize(z, p=2, dim=1)
        sim = torch.mm(z, z.T) / self.temperature

        # Make a positive example mask, with a value of 1 representing the position corresponding to the sim matrix. 
        # The two samples are positive examples of each other and are symmetric matrices. 
        positive_mask = np.eye(N=N, k=z_i.shape[0])
        positive_mask = torch.Tensor(positive_mask + positive_mask.T).to(self.device)

        self_mask = torch.ones((N, N)).fill_diagonal_(0).to(self.device)

        # calculate normalized cross entropy value
        loss = torch.mean(torch.log(torch.sum(torch.exp(sim)*self_mask, dim=1)) - (torch.sum(sim * positive_mask, dim=1)))

        return loss
