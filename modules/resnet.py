import torch
# from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
import torchvision.models.resnet as resnet_modules


# Class for generating out of the box resnets.
class ResNet(torch.nn.Module) :
    
    def __init__ (self, block, layers, num_class = 100, zero_init_residual = False,
                  groups = 1, width_per_group = 64, replace_stride_with_dilation=None,
                  norm_layer = None):
        
        super(ResNet, self).__init__()
        
        
        # Setting up normalization layer to batchnorm
        if norm_layer == None : 
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        # Setting up args for strides and dilations
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None : 
            # Indicates if we should replace 2x2 stride with
            # dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        # Bounds checking for replace_stride_with_dilation
        if len(replace_stride_with_dilation) != 3 : 
            raise ValueError("Incorrect Dimensions. replace_stride_with_dilation should "
                             "be of len 3 or None")
        
        
        
        # Setting up the model
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3,
                                     bias = False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate = replace_stride_with_dilation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.rep_dim = 512 * block.expansion
        
        # Initializing the weights with kaiming_uniform
        for module in self.modules() : 
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules() : 
                if isinstance(module, resnet_modules.Bottleneck):
                    torch.nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, resnet_modules.BasicBlock):
                    torch.nn.init.constant_(module.bn2.weight, 0)
                    
        
        
    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False) : 
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # Setting up downsampling
        if dilate : 
            self.dilate *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                resnet_modules.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * planes * block.expansion)
            )
            
        # Making the full ResNet Layer here.
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks) : 
            layers.append(block(self.inplanes, planes, groups, self.groups, 
                                base_width = self.base_width, dilation = self.dilation,
                                norm_layer = norm_layer))
            
        return torch.nn.Sequential(*layers)
        
    
    def _forward(self, x) : 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    
    def forward(self, X) : 
        return self._forward(X)
    
    
# Return the requested version of the     
def get_resnet(name) : 
    resnet = None
    if name == 'ResNet18' : 
        resnet = ResNet(block = resnet_modules.BasicBlock, layers = [2, 2, 2, 2])
    elif name == 'ResNet34' : 
        resnet = ResNet(block = resnet_modules.BasicBlock, layers = [3, 4, 6, 3])
    elif name == 'ResNet50' :
        resnet = ResNet(block = resnet_modules.Bottleneck, layers = [3, 4, 6, 3])
    else : 
        raise ValueError("This ResNet version not defined")

    return resnet