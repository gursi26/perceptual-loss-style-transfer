import torch 
import torch.nn.functional as F


# A single convolutional layer 
class Conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, upsample=False):
        super(Conv, self).__init__()
        self.upsample = upsample
        if upsample : 
            stride = 1 
        self.reflection_pad = torch.nn.ReflectionPad2d(kernel_size // 2) 
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x) : 
        if self.upsample : 
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.reflection_pad(x)
        x = self.conv(x)
        return x 


def test_conv():
    x = torch.randn((1, 3, 256, 256))
    same_size_layer = Conv(3, 3, 3, 1)
    half_size_layer = Conv(3, 3, 3, 2)
    double_size_layer = Conv(3, 3, 3, 1, upsample=True)
    print(f'Input shape : {x.shape}')
    print(f'Same size : {same_size_layer(x).shape}')
    print(f'Half size : {half_size_layer(x).shape}')
    print(f'Double size : {double_size_layer(x).shape}')

# test_conv()


# A single residual layer
class Residual(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_intermediate_layers=2):
        super(Residual, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_intermediate_layers):
            self.layers.append(Conv(in_channels, out_channels, 3, 1))
            self.layers.append(torch.nn.InstanceNorm2d(out_channels))
            if i < num_intermediate_layers - 1 : 
                self.layers.append(torch.nn.ReLU())

    def forward(self,x):
        identity = x.clone()
        for layer in self.layers : 
            x = layer(x)
        x = x + identity 
        return x       


def test_residual():
    x = torch.randn((1, 3, 256, 256))
    res_layer = Residual(3, 3)
    print(f'Input shape : {x.shape}')
    print(f'Residual : {res_layer(x).shape}')

# test_residual()


# Full network for style transfer
class ResNet(torch.nn.Module):

    def __init__(self, num_residual_layers=5):
        super(ResNet, self).__init__()
        # (in_channels, out_channels, kernel_size, stride)
        self.params_list = [(3,32,9,1),
                            (32,64,3,2),
                            (64,128,3,2)]

        self.downsample_block = torch.nn.ModuleList()
        for params in self.params_list : 
            self.downsample_block.append(Conv(
                in_channels=params[0], out_channels=params[1],
                kernel_size=params[2], stride=params[3]
                ))
            self.downsample_block.append(torch.nn.InstanceNorm2d(params[1]))
            self.downsample_block.append(torch.nn.ReLU())

        self.residual_block = torch.nn.ModuleList()
        for i in range(num_residual_layers):
            self.residual_block.append(Residual(self.params_list[-1][1], self.params_list[-1][1]))

        self.upsample_block = torch.nn.ModuleList()
        for params in reversed(self.params_list[1:]):
            self.upsample_block.append(Conv(
                in_channels=params[1], out_channels=params[0],
                kernel_size=params[2], stride=params[3], upsample=True
                ))
            self.upsample_block.append(torch.nn.InstanceNorm2d(params[0]))
            self.upsample_block.append(torch.nn.ReLU())
        
        self.output_layer = Conv(
            in_channels=self.params_list[0][1], out_channels=3,
            kernel_size=self.params_list[0][2], stride=self.params_list[0][3]
        )

    def forward(self, x):
        temp = [self.downsample_block, self.residual_block, self.upsample_block]
        for block in temp : 
            for layer in block : 
                x = layer(x) 
        x = self.output_layer(x)
        return x 

def test_resnet():
    # input shape - (batch_size, channels, height, width)
    x = torch.randn((1, 3, 256, 256))
    resnet = ResNet()
    print(f'Input shape : {x.shape}')
    print(f'ResNet : {resnet(x).shape}')

#test_resnet()