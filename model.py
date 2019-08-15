from torch import nn

import torch
import matplotlib.pyplot as plt
from dataset_factory import dataset_factory
import matplotlib2tikz
from pathloss_38901 import pathloss_38901
import numpy as np

class ImageModel(nn.Module):
    def __init__(self, channels, image_size, out_channels, kernel_size):
        super(ImageModel, self).__init__()

        # Uses BasicConvBlock which consists of
        # 2D convolution -> ReLU -> batchnorm -> maxpooling
        # Network consists of:
        # BasicConvBlock -> BasicConvBlock -> BasicConvBlock -> BasicConvBlock -> Linear

        self.blocks = nn.ModuleList()

    
        
        for idx, layer in enumerate(out_channels):
            if idx == 0:
                block = BasicConvBlock(channels, out_channels[0], (2,2), 0.2, kernel_size=kernel_size[idx], padding=2, stride=1)
                output_size = block.get_output_size(image_size)
            else:
                block = BasicConvBlock(out_channels[idx-1], out_channels[idx], (2,2), 0.1, kernel_size=kernel_size[idx])
                output_size = block.get_output_size(output_size)

            self.blocks.append(block)
            
        self.output_size = output_size[0]*output_size[1]
        self.Z = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        for block in self.blocks:
            x = block(x)
        x = x.view(batch_size, -1)
        x = self.Z(x)
        return x

class FeatureModel(nn.Module):
    def __init__(self, num_features, output_size, **kwargs):
        super(FeatureModel, self).__init__()

        # Dense neural network. Consists of:
        # Linear -> ReLU -> Batchnorm
        
        nn_layers = kwargs.get('nn_layers') if kwargs.get('nn_layers') else [200]
        self.dnn_layers = nn.ModuleList()

        self.dnn_layers.append(nn.Linear(num_features, nn_layers[0]))
        self.dnn_layers.append(nn.ReLU())
        self.dnn_layers.append(nn.BatchNorm1d(nn_layers[0]))

        if len(nn_layers) > 1:
            for layer_idx, layer in enumerate(nn_layers[1:]):
                self.dnn_layers.append(nn.Linear(nn_layers[layer_idx-1], layer))
                self.dnn_layers.append(nn.ReLU())
                self.dnn_layers.append(nn.BatchNorm1d(layer))
            
        self.output_layer = nn.Linear(nn_layers[-1], output_size)


    def forward(self, x):
        for module in self.dnn_layers:
            x = module(x)

        x = self.output_layer(x)
        return x

class SkynetModel(nn.Module):
    def __init__(self, args, **kwargs):
        super(SkynetModel, self).__init__()

        # Complete model. Consists of:
        # Distance  -> [Physics model]  ----------------------
        #                      |                             |
        # Feature   -> [Feature Model]  ->  Add -> [NN2] -> Add ----->
        #                                    |              
        # Image     -> [Image Model]   ------ 

        self.channels = args.channels
        num_features = args.num_features
        image_size = args.image_size
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        self.is_cuda = args.cuda
        self.offset_811 = args.offset_811
        self.offset_2630 = args.offset_2630

        self.model_mode = args.model_mode
        self.nn_layers = args.nn_layers

        self.rsrp_mu = torch.squeeze(torch.tensor(kwargs.get('rsrp_mu')).float())
        self.rsrp_std = torch.squeeze(torch.tensor(kwargs.get('rsrp_std')).float())
        self.image_output_size = 100


        if not self.model_mode == 'features-only':
            self.ImageModel = ImageModel(self.channels, image_size, out_channels, kernel_size)
            self.image_output_size = self.ImageModel.output_size
            if self.is_cuda:
                self.ImageModel = self.ImageModel.cuda()
            else:
                self.ImageModel = self.ImageModel.cpu()
        if not self.model_mode == 'images-only':
            self.FeatureModel = FeatureModel(num_features, self.image_output_size, nn_layers=self.nn_layers)
            if self.is_cuda:
                self.FeatureModel = self.FeatureModel.cuda()
            else:
                self.FeatureModel = self.FeatureModel.cpu()
            

        self.nn2 = nn.ModuleList()
        self.nn2.append(nn.Linear(self.image_output_size, 16))                
        self.nn2.append(nn.ReLU())
        self.nn2.append(nn.BatchNorm1d(16))
        self.nn2.append(nn.Linear(16, 1))

        if self.cuda:
            self = self.cuda()
        else:
            self = self.cpu()

    def forward(self, features, image, distance, **kwargs):
       
        P = self.predict_physicals_model(features, distance)
        P = P.detach()

        
        features_ = torch.cat([features, P],1) # Add computed pathloss to feature input

        tmp = 0
        if not self.model_mode == 'features-only':
            if not self.is_cuda:
                self.ImageModel = self.ImageModel.cpu()

            I = self.ImageModel(image)
            
            tmp += I
        if not self.model_mode == 'images-only':
            if not self.is_cuda:
                self.FeatureModel = self.FeatureModel.cpu()

            F = self.FeatureModel(features_)
            
            tmp += F
        
        correction = tmp

        for module in self.nn2:
            if self.is_cuda:
                module = module.cuda()
            else:
                module = module.cpu()
            correction = module(correction)

        if self.model_mode == 'data-driven':
            sum_out = correction  # fully data-driven thus not a correct of path loss
        else:
            sum_out = correction + P

        return correction, sum_out

    def predict_physicals_model(self, features, distance):
        frequency, offset = self.get_constants(features, distance)
        P = self.PhysicsModel(distance, frequency, offset=offset)
        if self.is_cuda:
            P = P.cuda()
        else:
            P = P.cpu()

        return P 


    def PhysicsModel(self, distance, frequency, offset, **kwargs):
        """

            P_tx: Transmission power of basestation (default 43 dBm)
            offset: Calibration offset (default 0)

            Uses 38.901 to compute path loss
            L(d) = P_tx - loss(d) + offset        

        """
        if self.cuda():
            distance = distance.cpu()
        loss = pathloss_38901(distance.numpy(), frequency.numpy())
        loss = torch.from_numpy(loss)

        
        P_tx = torch.tensor(43)
        
        # Tx gain
        P_rx = P_tx - loss + offset

        # RSRP conversion
        N = torch.tensor(100).float()
        P_rsrp = P_rx - 10*torch.log10(12*N)

        # Normalize 
        P_rsrp = (P_rsrp-self.rsrp_mu)/self.rsrp_std

        return P_rsrp

    def get_constants(self, features, distance):
        
        # Variables for physics model. Calibration and frequency.
        frequency = torch.empty((distance.shape))
        offset = torch.empty((distance.shape))
        frequency[features[:,7] == 1] = torch.tensor(2.63)
        frequency[features[:,7] != 1] = torch.tensor(0.811)
        offset[features[:,7] == 1] = torch.tensor(self.offset_2630)
        offset[features[:,7] != 1] = torch.tensor(self.offset_811)
        return frequency, offset

    def MSE_physicsmodel(self, distance, targets):
        frequency, offset = self.get_constants(features, distance)
        P = self.PhysicsModel(distance, frequency, offset=offset)
        MSE_norm = torch.mean(torch.sum(torch.abs(P-targets)**2),0)
        MSE = torch.mean(torch.sum(torch.abs((P*self.rsrp_std) + self.rsrp_mu -((targets*self.rsrp_std)  +self.rsrp_mu))**2),0)
        return MSE, MSE_norm




class BasicConvBlock(nn.Module):

    def __init__(self, channels, z_dim, max_pool, leaky_relu,  **kwargs):
        super(BasicConvBlock, self).__init__()
        self.kernel_size = kwargs.get('kernel_size') if kwargs.get('kernel_size') else (1,1)
        self.padding = kwargs.get('padding') if kwargs.get('padding') else 0
        self.stride = kwargs.get('stride') if kwargs.get('stride') else 1
        self.dilation = kwargs.get('dilation') if kwargs.get('dilation') else 1
        self.max_pool = kwargs.get('max_pool') if kwargs.get('max_pool') else (2,2)
    
        self.conv = nn.Conv2d(channels, z_dim, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu)
        self.batchnorm = nn.BatchNorm2d(z_dim)
        self.pool = nn.MaxPool2d(kernel_size=max_pool)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)
        return x

    def get_output_size(self, h_w):
        from math import floor
        if type(self.kernel_size) is not tuple:
            kernel_size = (self.kernel_size, self.kernel_size)
        h = floor( ((h_w[0] + (2 * self.padding) - ( self.dilation * (self.kernel_size[0] - 1) ) - 1 )/ self.stride) + 1)
        w = floor( ((h_w[1] + (2 * self.padding) - ( self.dilation * (self.kernel_size[1] - 1) ) - 1 )/ self.stride) + 1)
        size = [int(h/self.max_pool[0]), int(w/self.max_pool[1])]
        return size




def plot_physics_model():
    distance = np.linspace(10, 1600)
    

    train_dataset, valid_dataset, test_dataset = dataset_factory()

    num_features = train_dataset.features.shape[1]+1
    image_size = [256, 256]
    out_channels = [120, 60, 12, 6, 1]
    channels = 3
    rsrp_mu = train_dataset.target_mu
    rsrp_std = train_dataset.target_std
    model = SkynetModel(channels, num_features, image_size, out_channels, rsrp_mu = rsrp_mu, rsrp_std = rsrp_std)
    

    P_811 = model.PhysicsModel(distance, torch.tensor(0.811), offset=18) 
    P_811_unnorm = P_811.numpy()*rsrp_std+rsrp_mu
    P_2630 = model.PhysicsModel(distance, torch.tensor(2.630), offset=0)
    P_2630_unnorm = P_2630.numpy()*rsrp_std+rsrp_mu  

    mse_2630, mse_2630_norm = model.MSE_physicsmodel(train_dataset.distances, train_dataset.targets)
    print("MSE of 2630 MHz {}".format(mse_2630.numpy()))
    print("MSE of 2630 MHz {} (normalized)".format(mse_2630_norm.numpy()))

    with plt.style.context('seaborn'):
        fig = plt.figure(figsize=(6,4))
        plt.plot(train_dataset.distances[train_dataset.features[:,7] == 1]*1000, train_dataset.targets_unnorm[train_dataset.features[:,7] == 1],'o', label='Measurements 2630 MHz', markersize=3)
        plt.plot(train_dataset.distances[train_dataset.features[:,7] != 1]*1000, train_dataset.targets_unnorm[train_dataset.features[:,7] != 1],'o', label='Measurements 811 MHz', markersize=3)
    
        plt.plot(distance.numpy(), P_811_unnorm[0,:], label='UMa 811 MHz')
        plt.plot(distance.numpy(), P_2630_unnorm[0,:], label='UMa 2630 MHz')
        plt.xlabel('Distance [m]')
        plt.ylabel('RSRP [dBm]')
        plt.legend()
        plt.savefig("plots/rsrp_uma_measurements.eps")
        plt.show()
        plt.tight_layout()
        
    
if __name__ == '__main__':
    plot_physics_model()


