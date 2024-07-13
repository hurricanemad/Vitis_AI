# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0(ResNet::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/ret.3(ResNet::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/3464(ResNet::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #ResNet::ResNet/MaxPool2d[maxpool]/3479(ResNet::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/ret.7(ResNet::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/3508(ResNet::nndct_relu_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/ret.11(ResNet::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/3537(ResNet::nndct_elemwise_add_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/3538(ResNet::nndct_relu_8)
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/ret.15(ResNet::nndct_conv2d_9)
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/3566(ResNet::nndct_relu_10)
        self.module_11 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/ret.19(ResNet::nndct_conv2d_11)
        self.module_12 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/3595(ResNet::nndct_elemwise_add_12)
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/3596(ResNet::nndct_relu_13)
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/ret.23(ResNet::nndct_conv2d_14)
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/3624(ResNet::nndct_relu_15)
        self.module_16 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/ret.27(ResNet::nndct_conv2d_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.31(ResNet::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/3680(ResNet::nndct_elemwise_add_18)
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/3681(ResNet::nndct_relu_19)
        self.module_20 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/ret.35(ResNet::nndct_conv2d_20)
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/3709(ResNet::nndct_relu_21)
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/ret.39(ResNet::nndct_conv2d_22)
        self.module_23 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/3738(ResNet::nndct_elemwise_add_23)
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/3739(ResNet::nndct_relu_24)
        self.module_25 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/ret.43(ResNet::nndct_conv2d_25)
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/3767(ResNet::nndct_relu_26)
        self.module_27 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/ret.47(ResNet::nndct_conv2d_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.51(ResNet::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/3823(ResNet::nndct_elemwise_add_29)
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/3824(ResNet::nndct_relu_30)
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/ret.55(ResNet::nndct_conv2d_31)
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/3852(ResNet::nndct_relu_32)
        self.module_33 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/ret.59(ResNet::nndct_conv2d_33)
        self.module_34 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/3881(ResNet::nndct_elemwise_add_34)
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/3882(ResNet::nndct_relu_35)
        self.module_36 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]/ret.63(ResNet::nndct_conv2d_36)
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/3910(ResNet::nndct_relu_37)
        self.module_38 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]/ret.67(ResNet::nndct_conv2d_38)
        self.module_39 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.71(ResNet::nndct_conv2d_39)
        self.module_40 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/3966(ResNet::nndct_elemwise_add_40)
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/3967(ResNet::nndct_relu_41)
        self.module_42 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/ret.75(ResNet::nndct_conv2d_42)
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/3995(ResNet::nndct_relu_43)
        self.module_44 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]/ret.79(ResNet::nndct_conv2d_44)
        self.module_45 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/4024(ResNet::nndct_elemwise_add_45)
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/4025(ResNet::nndct_relu_46)
        self.module_47 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/4042(ResNet::nndct_adaptive_avg_pool2d_47)
        self.module_48 = py_nndct.nn.Module('nndct_flatten') #ResNet::ResNet/ret.83(ResNet::nndct_flatten_48)
        self.module_49 = py_nndct.nn.Linear(in_features=512, out_features=1000, bias=True) #ResNet::ResNet/Linear[fc]/ret(ResNet::nndct_dense_49)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(input=output_module_4, other=output_module_0, alpha=1)
        output_module_4 = self.module_8(output_module_4)
        output_module_9 = self.module_9(output_module_4)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_9 = self.module_12(input=output_module_9, other=output_module_4, alpha=1)
        output_module_9 = self.module_13(output_module_9)
        output_module_14 = self.module_14(output_module_9)
        output_module_14 = self.module_15(output_module_14)
        output_module_14 = self.module_16(output_module_14)
        output_module_17 = self.module_17(output_module_9)
        output_module_14 = self.module_18(input=output_module_14, other=output_module_17, alpha=1)
        output_module_14 = self.module_19(output_module_14)
        output_module_20 = self.module_20(output_module_14)
        output_module_20 = self.module_21(output_module_20)
        output_module_20 = self.module_22(output_module_20)
        output_module_20 = self.module_23(input=output_module_20, other=output_module_14, alpha=1)
        output_module_20 = self.module_24(output_module_20)
        output_module_25 = self.module_25(output_module_20)
        output_module_25 = self.module_26(output_module_25)
        output_module_25 = self.module_27(output_module_25)
        output_module_28 = self.module_28(output_module_20)
        output_module_25 = self.module_29(input=output_module_25, other=output_module_28, alpha=1)
        output_module_25 = self.module_30(output_module_25)
        output_module_31 = self.module_31(output_module_25)
        output_module_31 = self.module_32(output_module_31)
        output_module_31 = self.module_33(output_module_31)
        output_module_31 = self.module_34(input=output_module_31, other=output_module_25, alpha=1)
        output_module_31 = self.module_35(output_module_31)
        output_module_36 = self.module_36(output_module_31)
        output_module_36 = self.module_37(output_module_36)
        output_module_36 = self.module_38(output_module_36)
        output_module_39 = self.module_39(output_module_31)
        output_module_36 = self.module_40(input=output_module_36, other=output_module_39, alpha=1)
        output_module_36 = self.module_41(output_module_36)
        output_module_42 = self.module_42(output_module_36)
        output_module_42 = self.module_43(output_module_42)
        output_module_42 = self.module_44(output_module_42)
        output_module_42 = self.module_45(input=output_module_42, other=output_module_36, alpha=1)
        output_module_42 = self.module_46(output_module_42)
        output_module_42 = self.module_47(output_module_42)
        output_module_42 = self.module_48(input=output_module_42, start_dim=1, end_dim=-1)
        output_module_42 = self.module_49(output_module_42)
        return output_module_42