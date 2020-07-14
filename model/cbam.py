import torch, time
import torch.nn as nn
import torch.nn.functional as F

# "CBAM: Convolutional Block Attention Module", Woo et al., ECCV 2018
# ref. code: https://github.com/luuuyi/CBAM.PyTorch

class CBAModule(nn.Module):
    def __init__(self, fdim, reduce=16, use_bias=True):
        super(type(self), self).__init__()        
        # channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(fdim, fdim//reduce, 1, bias=use_bias)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(fdim//reduce, fdim, 1, bias=use_bias)
        # spatial attention branch
        self.conv3 = nn.Conv2d(2, 1, 5, 1, 2, bias=use_bias)
        self.sigmoid = nn.Sigmoid()
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0., 1e-3)
                nn.init.constant_(m.bias, 0.)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        
    def forward(self, x):
        # input feature x
        # channel augmentation stage
        avg_out = self.conv2( self.relu1( self.conv1( self.avg_pool(x) ) ) )
        max_out = self.conv2( self.relu1( self.conv1( self.max_pool(x) ) ) )
        ca_out = self.sigmoid( avg_out + max_out )
        x = x * ca_out
        # spatial augmentation stage
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        sp_out = torch.cat((avg_out,max_out), dim=1)
        sp_out = self.sigmoid( self.conv3(sp_out) )
        x = x * sp_out
        return x, (ca_out,sp_out)
        
        
        