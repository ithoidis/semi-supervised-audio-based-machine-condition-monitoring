import torch
import torch.nn as nn
import torch.nn.functional as F


class RawdNet(nn.Module):
    def __init__(self, input_shape, output_shape, emb_dim=25):
        super(M11_beam, self).__init__()
        self.upsampling_factor = 3
        self.length = input_shape[-1] * self.upsampling_factor
        self.channels = input_shape[0]
        self.context_win_size = 50
        self.spatial_filter = spatial_filter
        self.spatial_filter_size = 51
        self.flat_dim = 256
        self.emb_dim = emb_dim

        norm_layer = nn.GroupNorm # is layer normalization

        self.conv1 = nn.Conv1d(self.channels, 32, 81, 4, padding=81//2)  # 64 was input_shape[0]
        self.norm1 = norm_layer(1, 32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 32, 3, padding=3//2)
        self.conv21 = nn.Conv1d(32, 32, 3, padding=3//2)
        self.norm2 = norm_layer(1, 32)
        self.norm21 = norm_layer(1, 32)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(32, 64, 3, padding=3//2)  # 128
        self.conv31 = nn.Conv1d(64, 64, 3, padding=3//2)
        self.norm3 = norm_layer(1, 64)
        self.norm31 = norm_layer(1, 64)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(64, 128, 3, padding=3//2)  # 256
        self.norm4 = norm_layer(1, 128)
        self.conv41 = nn.Conv1d(128, 128, 3, padding=3//2)
        self.norm41 = norm_layer(1, 128)
        self.conv42 = nn.Conv1d(128, 128, 3, padding=3//2)
        self.norm42 = norm_layer(1, 128)
        self.pool4 = nn.MaxPool1d(4)

        self.conv5 = nn.Conv1d(128, 256, 3, padding=3//2)  # 512
        self.norm5 = norm_layer(1, 256)
        self.conv51 = nn.Conv1d(256, 256, 3, padding=3//2)
        self.norm51 = norm_layer(1, 256)

        self.drop1 = nn.Dropout(0.2)
        self.fccenter = nn.Linear(self.flat_dim, self.emb_dim)
        self.drop2 = nn.Dropout(0.2)
        self.fcout = nn.Linear(self.emb_dim, output_shape)

    def forward(self, x):
            
            x = F.relu(self.norm1(self.conv1(x)))
            x = self.pool1(x)

            x = F.relu(self.norm2(self.conv2(x)))
            x = F.relu(self.norm21(self.conv21(x)))
            x = self.pool2(x)

            x = F.relu(self.norm3(self.conv3(x)))
            x = F.relu(self.norm31(self.conv31(x)))
            x = self.pool3(x)

            x = F.relu(self.norm4(self.conv4(x)))
            x = F.relu(self.norm41(self.conv41(x)))
            x = F.relu(self.norm42(self.conv42(x)))
            x = self.pool4(x)

            x = F.relu(self.norm5(self.conv5(x)))
            x = F.relu(self.norm51(self.conv51(x)))
            
            x = torch.mean(x, dim=-1)
            x = self.drop1(x)
            emb = self.fccenter(x)
            x = self.drop2(emb)
            x = self.fcout(x)
            x = F.log_softmax(x, dim=-1)
            return emb, x



class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.register_parameter('centers', self.centers) # no need to register manually. See nn.Module.__setattr__(...)
        self.use_cuda = False

    def forward(self, y, feat):
        # torch.histc can only be implemented on CPU
        # To calculate the total number of every class in one mini-batch. See Equation 4 in the paper
        if self.use_cuda:
            hist = Variable(torch.histc(y.cpu().data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1).cuda()
        else:
            hist = Variable(torch.histc(y.data.float(),bins=self.num_classes,min=0,max=self.num_classes) + 1)

        centers_count = hist.index_select(0,y.long())

        # To squeeze the Tenosr
        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))

        centers_pred = self.centers.index_select(0, y.long())    
        diff = (feat - centers_pred).pow(2).sum(1)
        loss = self.loss_weight * 1 / 2.0 * (diff / centers_count).sum()
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))



class DOC(nn.Module):
    def __init__(self, input_features, out_features=16, units=32):
        super(DOC, self).__init__()
        self.lr = 1e-3
        self.lr_svdd = 0.5
        self.out_features = out_features
        self.fc1 = nn.Linear(input_features, int(input_features/2), bias=False)
        self.fc2 = nn.Linear(int(input_features/2), units, bias=False)
        self.fc3 = nn.Linear(units, units, bias=False)
        self.fcout = nn.Linear(units, out_features, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fcout(x)


class SVDD(nn.Module):
    def __init__(self, feat_dim, loss_weight=1.0):
        super(SVDD, self).__init__()
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.center = nn.Parameter(torch.randn(feat_dim))
        self.radius = nn.Parameter(torch.randn(1))
        self.use_cuda = False

    def forward(self, y, feat):

        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()

        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))
       
        diff = (feat - self.centers.index_select(0, y.long())).pow(2).sum(1) - self.radius.pow(2)
        loss =  (torch.relu(diff) / y.shape[0]).sum()
        return self.radius.pow(2) + self.loss_weight * loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
