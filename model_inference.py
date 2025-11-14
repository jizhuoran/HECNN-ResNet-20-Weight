# Aespa model inference on Cifar-10
import os
from typing import  Callable, Optional
import torch
import torch.nn as nn
import copy
from math import pi, sqrt
from torch import Tensor
from torch.autograd import Function
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 3x3 convolution kernel
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
# 1x1 convolution kernel
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# HerPN activation function
class HerPN2d(nn.Module):
    """
    Three Hermite polynomial bases hi(x): 1, x, x^2 - 1, with normalization coefficients 1, 1, 0.707
    Corresponding f coefficients: 1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi)
    """
    @staticmethod
    def h0(x):
        return torch.ones(x.shape).to(x.device)

    @staticmethod
    def h1(x):
        return x

    @staticmethod
    def h2(x):
        return (x * x - 1)  * 0.7071

    def __init__(self, num_features: int, BN_dimension=2, BN_copy: nn.Module = None):
        super().__init__()
        self.f = (1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi))
        # self.f = ( 1 / 2, 1 / sqrt(4 * pi))
        self.num_channels = num_features
        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(self.num_channels),requires_grad=True)  # shape: (num_channels,)
        self.beta = nn.Parameter(torch.zeros(self.num_channels),requires_grad=True)  # shape: (num_channels,)

        if (BN_copy):
            self.bn0 = copy.deepcopy(BN_copy)
            self.bn1 = copy.deepcopy(BN_copy)
            self.bn2 = copy.deepcopy(BN_copy)

        elif (BN_dimension == 1):
            self.bn0 = nn.BatchNorm1d(num_features, affine=False)
            self.bn1 = nn.BatchNorm1d(num_features, affine=False)
            self.bn2 = nn.BatchNorm1d(num_features, affine=False)
        else:
            self.bn0 = nn.BatchNorm2d(num_features, affine=False)
            self.bn1 = nn.BatchNorm2d(num_features, affine=False)
            self.bn2 = nn.BatchNorm2d(num_features, affine=False)

        self.bn = (self.bn0, self.bn1, self.bn2)
        # self.bn = ( self.bn1, self.bn2)
        self.h = (self.h0, self.h1, self.h2)
        # self.h = (self.h1, self.h2)

    def forward(self, x):
        result = torch.zeros(x.shape).to(x.device)

        for bn, f, h in zip(self.bn, self.f, self.h):
            temp = h(x)
            temp = bn(temp)
            temp = torch.mul(f, temp)
            result = torch.add(result, temp)
        result = self.gamma.view(1, -1, 1, 1) * result + self.beta.view(1, -1, 1, 1)
        return result
# ResNet Block and HerPN
class BasicBlock_HerPN(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.HerPN1 = HerPN2d(num_features=planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.HerPN2 = HerPN2d(num_features=planes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.HerPN1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.HerPN2(out)
        return out
# ResNet20 model
class ResNet20_HerPN(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResNet20_HerPN, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.HerPN1 = HerPN2d(16)
        self.layer1 = self._make_layer(block, 16, 3)
        self.layer2 = self._make_layer(block, 32, 3, stride=2)
        self.layer3 = self._make_layer(block, 64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.HerPN1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Get model
def get_resnet20_HerPN(num_classes):
    return ResNet20_HerPN(block=BasicBlock_HerPN,num_classes=num_classes)
# Convert a HerPN to a mathematically equivalent multi-channel quadratic polynomial
def change_One_HerPN2d(model:HerPN2d):
    """
    This function converts a HerPN2d into a multi-channel PAF:
    For the three bases, h0 can be completely removed (its output is only 1e-5); h1 and h2 are processed normally.
    :param model:
    :return:
    """
    bn1 = model.bn1
    bn2 = model.bn2
    gamma = model.gamma
    beta=model.beta
    var2 = (bn2.running_var + 1e-05)**-0.5
    var1 = (bn1.running_var + 1e-05)**-0.5
    u2 = bn2.running_mean
    u1 = bn1.running_mean
    w2 = gamma * var2 / sqrt(4 * pi)
    w1 = gamma * var1 * 0.5
    a2 = 0.5 * sqrt(2) * w2
    a1 = w1
    a0 = beta - 0.5 * sqrt(2) * w2 - u2 * w2 - w1 * u1
    new_model = MultiChannelPAF(a2, a1, a0)
    return new_model
# Convert all HerPN in the model to multi-channel quadratic polynomials
def change_all_HerPN(model):
    # Get a copy of the model
    model_modules = list(model.named_modules())
    # Find corresponding modules
    for name, module in model_modules:
        if isinstance(module, HerPN2d):
            # Check whether the current module is directly attached to model (i.e., module name is directly 'relu1', 'relu2', etc.)
            if name in model._modules:
                # Directly replace the attribute in model
                # new activation to replace with
                new_act = change_One_HerPN2d(module)
                setattr(model, name, new_act)  # Replace with the new activation
            else:
                # new activation to replace with
                new_act = change_One_HerPN2d(module)
                # For layer / block substructures
                parent_name = name.rsplit('.', 1)[0]
                parent = dict(model.named_modules())[parent_name]  # Get parent module
                # Delete the original ReLU layer in the parent module (if it exists)
                if hasattr(parent, name.rsplit('.', 1)[-1]):
                    delattr(parent, name.rsplit('.', 1)[-1])
                # Replace with the new Chebyshev_Relu_MaxScale
                setattr(parent, name.rsplit('.', 1)[-1], new_act)  # Replace with new activation
    return model
# Autograd function
class MultiChannelPoloActFunction(Function):
    @staticmethod
    def forward(ctx, input, a2, a1, a0):
        """
        Forward pass: compute y = a2 * x^2 + a1 * x + a0
        Args:
            input: input tensor, shape (batch_size, num_channels, height, width)
            a2, a1, a0: parameter tensors, shape (num_channels,)
        Returns:
            Output tensor with the same shape as input
        """
        # Save input and parameters for use in backprop
        ctx.save_for_backward(input, a2, a1, a0)

        # Expand a2, a1, a0 to match the shape of input
        a2 = a2.view(1, -1, 1, 1).to('cuda') # Shape becomes (1, num_channels, 1, 1)
        a1 = a1.view(1, -1, 1, 1).to('cuda') # Shape becomes (1, num_channels, 1, 1)
        a0 = a0.view(1, -1, 1, 1).to('cuda')  # Shape becomes (1, num_channels, 1, 1)
        # print(a2.device, a1.device, a0.device,input.device)

        # Compute forward propagation
        output = a2 * input.pow(2) + a1 * input + a0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients
        Args:
            grad_output: gradient of the output, same shape as input
        Returns:
            grad_input, grad_a2, grad_a1, grad_a0
        """
        # Retrieve saved tensors
        input, a2, a1, a0 = ctx.saved_tensors

        # Expand a2, a1, a0 to match the shape of input
        a2 = a2.view(1, -1, 1, 1)  # Shape becomes (1, num_channels, 1, 1)
        a1 = a1.view(1, -1, 1, 1)  # Shape becomes (1, num_channels, 1, 1)

        # Compute gradients
        grad_input = grad_output * (2 * a2 * input + a1)  # dL/dx = grad_output * (2*a2*x + a1)

        # Sum the gradients for a2, a1, a0 over batch, height, width dimensions
        grad_a2 = (grad_output * input.pow(2)).sum(dim=(0, 2, 3))  # dL/da2 = sum(grad_output * x^2)
        grad_a1 = (grad_output * input).sum(dim=(0, 2, 3))  # dL/da1 = sum(grad_output * x)
        grad_a0 = grad_output.sum(dim=(0, 2, 3))  # dL/da0 = sum(grad_output)

        return grad_input, grad_a2, grad_a1, grad_a0
# Multi-channel
class MultiChannelPAF(nn.Module):
    def __init__(self,init_a2, init_a1, init_a0):
        """
        Multi-channel second-order activation function module
        Args:
            num_channels: number of channels
            init_a2, init_a1, init_a0: initial values of the parameters, default 1.0, 0.0, 0.0
        """
        super(MultiChannelPAF, self).__init__()
        # Define a2, a1, a0 as learnable parameters
        self.a2 = init_a2
        self.a1 = init_a1
        self.a0 = init_a0
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor, shape (batch_size, num_channels, height, width)
        Returns:
            Output tensor with the same shape as x
        """
        return MultiChannelPoloActFunction.apply(x, self.a2, self.a1, self.a0)
#
def get_model():
    model = get_resnet20_HerPN(num_classes=10)
    model_path = './ResNet20_AESPA.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    model = change_all_HerPN(model)
    model.eval()
    return model
# Dataset
def get_Cifar10_dataloader():
    batch_size = 64
    test_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2470, 0.2435, 0.2616]),
                                         ])
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        # Random horizontal flip (probability 50%)
        transforms.RandomHorizontalFlip(p=0.5),
        # Random rotation: angle range [-30°, 30°]
        transforms.RandomRotation(degrees=30),
        # Convert to tensor and normalize (ImageNet standard)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    dateset_path = os.path.join('./data','cifar10')
    training_data = datasets.CIFAR10(root=dateset_path, train=True, download=True, transform=train_transform, )
    testing_data = datasets.CIFAR10(root=dateset_path, train=False, download=True, transform=test_transform, )
    # Training dataset and test dataset
    train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return train_data, test_data
# Inference
# def test(model):
#     _,test_loader = get_Cifar10_dataloader()
#     testing_correct = 0
#     test_loss = 0
#     # Switch to test mode
#     model.eval()
#     device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     loss_fun = nn.CrossEntropyLoss()
#     for x_test, y_test in test_loader:
#         x_test, y_test = x_test.to(device), y_test.to(device)
#         outputs = model(x_test)
#         loss = loss_fun(outputs, y_test)
#         _, pred = torch.max(outputs.data, 1)
#         testing_correct += torch.sum(pred == y_test.data)
#         test_loss += loss.item()
#
#     test_loss /= len(test_loader.dataset)
#     testing_correct = 100 * testing_correct / len(test_loader.dataset)
#     print(" Test Loss is:{:.4f} Test Accuracy is:{:.4f}%".format(test_loss ,testing_correct))
def test(model):
    _, test_loader = get_Cifar10_dataloader()
    testing_correct = 0
    test_loss = 0
    model.eval()  # Switch to eval mode (important: freeze BatchNorm/ Dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fun = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.float()  # Force cast to float32 (avoid integer overflow in computations)
            x_test, y_test = x_test.to(device), y_test.to(device)
            # 2. Check if input data contains NAN/inf (extreme case: data loading exception)
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f"警告：第 {batch_idx} 个batch的输入数据包含 NAN/inf！")
                continue  # Skip abnormal batch
            # 3. Model inference
            outputs = model(x_test)
            # 4. Check if model outputs contain NAN/inf (key to locating NAN source)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                # print(f"Warning: output of batch {batch_idx} contains NAN/inf!")
                # print(f"Output stats: mean={outputs.mean().item()}, max={outputs.max().item()}, min={outputs.min().item()}")
                continue  # Skip abnormal output to avoid contaminating loss/acc calculation

            # 5. Compute loss and accuracy (fix loss calculation logic)
            loss = loss_fun(outputs, y_test)
            _, pred = torch.max(outputs.data, 1)

            #
            testing_correct += torch.sum(pred == y_test.data).item()
            test_loss += loss.item()


    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * testing_correct / len(test_loader.dataset)
    print(f"Test Loss is:{avg_test_loss:.4f} Test Accuracy is:{test_acc:.4f}%")
    return avg_test_loss, test_acc
def main():
    model = get_model()
    test(model)
if __name__ == '__main__':
    main()
