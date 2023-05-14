import torch
import torch.nn as nn


class TSFE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, encoder, n_features,
                 patch_dim=(2, 2), normalize=True, projection_dim=128, gn=2):
        """
        :param n_features: the number of output channels set as the output feature of the encoder
        :param patch_dim : total number of patch, patch_dim[0]*patch_dim[1]  = num_patches
        :param normalize : whether regularization projection is required
        :projection_dim : the dimension of the features output by the final projector
        """
        super(TSFE_model, self).__init__()

        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-gn])
        self.n_features = n_features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # total number of patch:   patch_dim[0]*patch_dim[1]  = num_patches
        self.patch_dim = patch_dim

        # After adaptive pooling, the output size on each channel is patch_dim (as a tuple), 
        # the number of channels remains unchanged before and after
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)  # (2,2)

        # MLP for projector
        self.projector = nn.Sequential(
            # MLP1(n_features, n_features)
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            # MLP2(n_features, projection_dim)
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x_i, x_j):
        """
        :param x_i: full scale
        :param x_j: half scale
         ∈(b, c, M, M)
        """

        # global features
        # Full scale image input encoder to obtain feature y1, (b, n_features, 8,8)
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # local features
        # (,patch_dim[0],patch_dim[1])-> (b, n_features, patch_dim[0], patch_dim[1])
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)

        # get a feature in shape (n_features,), merge the last two dimensions to obtain num_patches, (b, n_features, num_patches)
        h_i_patch = h_i_patch.reshape(-1, self.n_features,
                                      self.patch_dim[0]*self.patch_dim[1])

        h_j_patch = h_j_patch.reshape(-1, self.n_features,
                                      self.patch_dim[0]*self.patch_dim[1])

        # Convert the last two dimensions -> Convert n_features placed at the back
        # (b, num_patches, n_features)
        h_i_patch = torch.transpose(h_i_patch, 2, 1)
        # (b* num_patches, n_features) -> get local features for all patches
        h_i_patch = h_i_patch.reshape(-1, self.n_features)

        h_j_patch = torch.transpose(h_j_patch, 2, 1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)

        # global dimension process, input projector with (,n_features)
        # batch_size and channel never change, average pooling of feature maps into one number, output dimension (b, n_features, 1,1)
        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)  # output (b, n_features, 1,1)

        # reshape the last dim as n_features, (b, n_features)
        h_i = h_i.view(-1, self.n_features)
        h_j = h_j.view(-1, self.n_features)

        if self.normalize:  # regularization
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)

            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)

        # global projections
        z_i = self.projector(h_i)  # (b, projector_dim)
        z_j = self.projector(h_j)

        # local projections
        # （b* num_patches, projector_dim）
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)

        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch
