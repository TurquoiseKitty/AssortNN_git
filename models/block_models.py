import torch.nn as nn
import torch
import numpy as np

def gen_sequential(
    channels,
    Norm_feature = 0,
    force_zero = False,
    force_zero_ceiling = 0.01,
    **kwargs
):

    

    assert len(channels) > 0

    modulist = []

    from_channel = channels[0]

    mid_layers = channels[1:]

    for layer in range(len(mid_layers)):

        to_channel = mid_layers[layer]
        
        linear_layer = nn.Linear(from_channel, to_channel)

        
        if force_zero:
            with torch.no_grad():
                linear_layer.bias.data.fill_(0)
                linear_layer.weight.uniform_(0, force_zero_ceiling)
            linear_layer.requires_grad = True
        
        modulist.append(linear_layer)

        if Norm_feature == 0:
            modulist.append(nn.BatchNorm1d(to_channel))

        else:
            modulist.append(nn.BatchNorm1d(Norm_feature))

        modulist.append(nn.LeakyReLU(0.2, inplace=True))

        from_channel = to_channel

    return nn.ModuleList(modulist)


class PartResBlock(nn.Module):

    def __init__(self, 
        channels,
        force_zero = False,
        prob_ending = False,
    ):

        super(PartResBlock, self).__init__()
        self.channels = channels
        self.body = gen_sequential(channels, force_zero=force_zero)
        self.prob_ending = prob_ending

    def forward(self, X, ResPos):
        # we only residual part of the input
        assert self.channels[-1] == ResPos
        previous_results = X[:, :ResPos]
        Y = X
        for m in self.body:
            Y = m(Y)

        if self.prob_ending:
            return nn.Softmax(-1)(torch.log(previous_results) + Y)
        else:
            return Y + previous_results


class FeatureEncoder(nn.Module):

    def __init__(self,
        Nprod_Veclen = 5,

        prod_normalize = False,
        Len_prodFeature = 3,
        cus_normalize = False,
        Len_customerFeature = 16,

        Num_cusEncoder_midLayer = 0,
        cusEncoder_midLayers = [],

        Num_prodEncoder_midLayer = 0,
        prodEncoder_midLayers = [],

        CROSS = False
    ):

        super(FeatureEncoder, self).__init__()

        self.Nprod_Veclen = Nprod_Veclen

        self.prod_normalize = prod_normalize
        self.Len_prodFeature = Len_prodFeature
        self.cus_normalize = cus_normalize
        self.Len_customerFeature = Len_customerFeature

        if self.prod_normalize:
            self.prod_bn1 = nn.BatchNorm1d(self.Nprod_Veclen * self.Len_prodFeature)

        if self.cus_normalize:
            self.cus_b1 = nn.BatchNorm1d(self.Len_customerFeature)

        self.CROSS = CROSS

        assert Num_cusEncoder_midLayer == len(cusEncoder_midLayers)

        self.cusEncoder_channels = np.insert(cusEncoder_midLayers, 0, self.Len_customerFeature)

        self.cusEncoder = gen_sequential(self.cusEncoder_channels)

        assert Num_prodEncoder_midLayer == len(prodEncoder_midLayers)

        if CROSS:

            assert len(prodEncoder_midLayers) == 0 

            self.prodEncoder_channels = np.insert(prodEncoder_midLayers, 0, Nprod_Veclen * self.Len_prodFeature)

            self.prodEncoder = gen_sequential(self.prodEncoder_channels)

        else:

            self.prodEncoder_channels = np.insert(prodEncoder_midLayers, 0, self.Len_prodFeature)

            self.prodEncoder = gen_sequential(self.prodEncoder_channels, Norm_feature=self.Nprod_Veclen)

        assert self.prodEncoder_channels[-1] == self.cusEncoder_channels[-1]


    def forward(self, IN):

        cusFs = IN[ : , : self.Len_customerFeature]
        prodFs = IN[ :, self.Len_customerFeature : ]

        if self.cus_normalize:

            cusFs = self.cus_b1(cusFs)

        if self.prod_normalize:

            prodFs = self.prod_bn1(prodFs)

        for m in self.cusEncoder:
            cusFs = m(cusFs)    

        prodFs = torch.reshape(prodFs, (len(prodFs), self.Nprod_Veclen, -1))

        if not self.CROSS:
            
            for m in self.prodEncoder:

                prodFs = m(prodFs)

            cusFs = torch.unsqueeze(cusFs, 2)

            fake_utils = torch.matmul(prodFs, cusFs)
            fake_utils = torch.squeeze(fake_utils, -1)

        else:

            paras = torch.reshape(cusFs, (len(prodFs), self.Nprod_Veclen, -1))

            fake_utils = (paras * prodFs).sum(dim=-1, keepdim=False)

        return fake_utils




        

