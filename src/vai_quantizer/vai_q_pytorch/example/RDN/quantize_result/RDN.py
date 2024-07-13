# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class RDN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(RDN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #RDN::input_0(RDN::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Conv2d[sfe1]/ret.3(RDN::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Conv2d[sfe2]/ret.5(RDN::nndct_conv2d_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.7(RDN::nndct_conv2d_3)
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5061(RDN::nndct_relu_4)
        self.module_5 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[0]/ret.9(RDN::nndct_concat_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.11(RDN::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5086(RDN::nndct_relu_7)
        self.module_8 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[1]/ret.13(RDN::nndct_concat_8)
        self.module_9 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.15(RDN::nndct_conv2d_9)
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5111(RDN::nndct_relu_10)
        self.module_11 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[0]/Sequential[layers]/DenseLayer[2]/ret.17(RDN::nndct_concat_11)
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[0]/Conv2d[lff]/ret.19(RDN::nndct_conv2d_12)
        self.module_13 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[0]/ret.21(RDN::nndct_elemwise_add_13)
        self.module_14 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.23(RDN::nndct_conv2d_14)
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5159(RDN::nndct_relu_15)
        self.module_16 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[0]/ret.25(RDN::nndct_concat_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.27(RDN::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5184(RDN::nndct_relu_18)
        self.module_19 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[1]/ret.29(RDN::nndct_concat_19)
        self.module_20 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.31(RDN::nndct_conv2d_20)
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5209(RDN::nndct_relu_21)
        self.module_22 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[1]/Sequential[layers]/DenseLayer[2]/ret.33(RDN::nndct_concat_22)
        self.module_23 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[1]/Conv2d[lff]/ret.35(RDN::nndct_conv2d_23)
        self.module_24 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[1]/ret.37(RDN::nndct_elemwise_add_24)
        self.module_25 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.39(RDN::nndct_conv2d_25)
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5257(RDN::nndct_relu_26)
        self.module_27 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[0]/ret.41(RDN::nndct_concat_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.43(RDN::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5282(RDN::nndct_relu_29)
        self.module_30 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[1]/ret.45(RDN::nndct_concat_30)
        self.module_31 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.47(RDN::nndct_conv2d_31)
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5307(RDN::nndct_relu_32)
        self.module_33 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[2]/Sequential[layers]/DenseLayer[2]/ret.49(RDN::nndct_concat_33)
        self.module_34 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[2]/Conv2d[lff]/ret.51(RDN::nndct_conv2d_34)
        self.module_35 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[2]/ret.53(RDN::nndct_elemwise_add_35)
        self.module_36 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.55(RDN::nndct_conv2d_36)
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5355(RDN::nndct_relu_37)
        self.module_38 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[0]/ret.57(RDN::nndct_concat_38)
        self.module_39 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.59(RDN::nndct_conv2d_39)
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5380(RDN::nndct_relu_40)
        self.module_41 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[1]/ret.61(RDN::nndct_concat_41)
        self.module_42 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.63(RDN::nndct_conv2d_42)
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5405(RDN::nndct_relu_43)
        self.module_44 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[3]/Sequential[layers]/DenseLayer[2]/ret.65(RDN::nndct_concat_44)
        self.module_45 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[3]/Conv2d[lff]/ret.67(RDN::nndct_conv2d_45)
        self.module_46 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[3]/ret.69(RDN::nndct_elemwise_add_46)
        self.module_47 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.71(RDN::nndct_conv2d_47)
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5453(RDN::nndct_relu_48)
        self.module_49 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[0]/ret.73(RDN::nndct_concat_49)
        self.module_50 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.75(RDN::nndct_conv2d_50)
        self.module_51 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5478(RDN::nndct_relu_51)
        self.module_52 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[1]/ret.77(RDN::nndct_concat_52)
        self.module_53 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.79(RDN::nndct_conv2d_53)
        self.module_54 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5503(RDN::nndct_relu_54)
        self.module_55 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[4]/Sequential[layers]/DenseLayer[2]/ret.81(RDN::nndct_concat_55)
        self.module_56 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[4]/Conv2d[lff]/ret.83(RDN::nndct_conv2d_56)
        self.module_57 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[4]/ret.85(RDN::nndct_elemwise_add_57)
        self.module_58 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.87(RDN::nndct_conv2d_58)
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5551(RDN::nndct_relu_59)
        self.module_60 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[0]/ret.89(RDN::nndct_concat_60)
        self.module_61 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.91(RDN::nndct_conv2d_61)
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5576(RDN::nndct_relu_62)
        self.module_63 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[1]/ret.93(RDN::nndct_concat_63)
        self.module_64 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.95(RDN::nndct_conv2d_64)
        self.module_65 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5601(RDN::nndct_relu_65)
        self.module_66 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[5]/Sequential[layers]/DenseLayer[2]/ret.97(RDN::nndct_concat_66)
        self.module_67 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[5]/Conv2d[lff]/ret.99(RDN::nndct_conv2d_67)
        self.module_68 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[5]/ret.101(RDN::nndct_elemwise_add_68)
        self.module_69 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.103(RDN::nndct_conv2d_69)
        self.module_70 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5649(RDN::nndct_relu_70)
        self.module_71 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[0]/ret.105(RDN::nndct_concat_71)
        self.module_72 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.107(RDN::nndct_conv2d_72)
        self.module_73 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5674(RDN::nndct_relu_73)
        self.module_74 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[1]/ret.109(RDN::nndct_concat_74)
        self.module_75 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.111(RDN::nndct_conv2d_75)
        self.module_76 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5699(RDN::nndct_relu_76)
        self.module_77 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[6]/Sequential[layers]/DenseLayer[2]/ret.113(RDN::nndct_concat_77)
        self.module_78 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[6]/Conv2d[lff]/ret.115(RDN::nndct_conv2d_78)
        self.module_79 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[6]/ret.117(RDN::nndct_elemwise_add_79)
        self.module_80 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[0]/Conv2d[conv]/ret.119(RDN::nndct_conv2d_80)
        self.module_81 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[0]/ReLU[relu]/5747(RDN::nndct_relu_81)
        self.module_82 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[0]/ret.121(RDN::nndct_concat_82)
        self.module_83 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[1]/Conv2d[conv]/ret.123(RDN::nndct_conv2d_83)
        self.module_84 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[1]/ReLU[relu]/5772(RDN::nndct_relu_84)
        self.module_85 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[1]/ret.125(RDN::nndct_concat_85)
        self.module_86 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[2]/Conv2d[conv]/ret.127(RDN::nndct_conv2d_86)
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[2]/ReLU[relu]/5797(RDN::nndct_relu_87)
        self.module_88 = py_nndct.nn.Cat() #RDN::RDN/RDB[rdbs]/ModuleList[7]/Sequential[layers]/DenseLayer[2]/ret.129(RDN::nndct_concat_88)
        self.module_89 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/RDB[rdbs]/ModuleList[7]/Conv2d[lff]/ret.131(RDN::nndct_conv2d_89)
        self.module_90 = py_nndct.nn.Add() #RDN::RDN/RDB[rdbs]/ModuleList[7]/ret.133(RDN::nndct_elemwise_add_90)
        self.module_91 = py_nndct.nn.Cat() #RDN::RDN/ret.135(RDN::nndct_concat_91)
        self.module_92 = py_nndct.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Sequential[gff]/Conv2d[0]/ret.137(RDN::nndct_conv2d_92)
        self.module_93 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Sequential[gff]/Conv2d[1]/ret.139(RDN::nndct_conv2d_93)
        self.module_94 = py_nndct.nn.Add() #RDN::RDN/ret.141(RDN::nndct_elemwise_add_94)
        self.module_95 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Sequential[upscale]/Conv2d[0]/ret.143(RDN::nndct_conv2d_95)
        self.module_96 = py_nndct.nn.Module('nndct_pixel_shuffle',upscale_factor=2) #RDN::RDN/Sequential[upscale]/PixelShuffle[1]/ret.145(RDN::nndct_pixel_shuffle_96)
        self.module_97 = py_nndct.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #RDN::RDN/Conv2d[output]/ret(RDN::nndct_conv2d_97)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_2 = self.module_2(output_module_0)
        output_module_3 = self.module_3(output_module_2)
        output_module_3 = self.module_4(output_module_3)
        output_module_5 = self.module_5(dim=1, tensors=[output_module_2,output_module_3])
        output_module_6 = self.module_6(output_module_5)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(dim=1, tensors=[output_module_5,output_module_6])
        output_module_9 = self.module_9(output_module_8)
        output_module_9 = self.module_10(output_module_9)
        output_module_11 = self.module_11(dim=1, tensors=[output_module_8,output_module_9])
        output_module_11 = self.module_12(output_module_11)
        output_module_13 = self.module_13(input=output_module_2, other=output_module_11, alpha=1)
        output_module_14 = self.module_14(output_module_13)
        output_module_14 = self.module_15(output_module_14)
        output_module_16 = self.module_16(dim=1, tensors=[output_module_13,output_module_14])
        output_module_17 = self.module_17(output_module_16)
        output_module_17 = self.module_18(output_module_17)
        output_module_19 = self.module_19(dim=1, tensors=[output_module_16,output_module_17])
        output_module_20 = self.module_20(output_module_19)
        output_module_20 = self.module_21(output_module_20)
        output_module_22 = self.module_22(dim=1, tensors=[output_module_19,output_module_20])
        output_module_22 = self.module_23(output_module_22)
        output_module_24 = self.module_24(input=output_module_13, other=output_module_22, alpha=1)
        output_module_25 = self.module_25(output_module_24)
        output_module_25 = self.module_26(output_module_25)
        output_module_27 = self.module_27(dim=1, tensors=[output_module_24,output_module_25])
        output_module_28 = self.module_28(output_module_27)
        output_module_28 = self.module_29(output_module_28)
        output_module_30 = self.module_30(dim=1, tensors=[output_module_27,output_module_28])
        output_module_31 = self.module_31(output_module_30)
        output_module_31 = self.module_32(output_module_31)
        output_module_33 = self.module_33(dim=1, tensors=[output_module_30,output_module_31])
        output_module_33 = self.module_34(output_module_33)
        output_module_35 = self.module_35(input=output_module_24, other=output_module_33, alpha=1)
        output_module_36 = self.module_36(output_module_35)
        output_module_36 = self.module_37(output_module_36)
        output_module_38 = self.module_38(dim=1, tensors=[output_module_35,output_module_36])
        output_module_39 = self.module_39(output_module_38)
        output_module_39 = self.module_40(output_module_39)
        output_module_41 = self.module_41(dim=1, tensors=[output_module_38,output_module_39])
        output_module_42 = self.module_42(output_module_41)
        output_module_42 = self.module_43(output_module_42)
        output_module_44 = self.module_44(dim=1, tensors=[output_module_41,output_module_42])
        output_module_44 = self.module_45(output_module_44)
        output_module_46 = self.module_46(input=output_module_35, other=output_module_44, alpha=1)
        output_module_47 = self.module_47(output_module_46)
        output_module_47 = self.module_48(output_module_47)
        output_module_49 = self.module_49(dim=1, tensors=[output_module_46,output_module_47])
        output_module_50 = self.module_50(output_module_49)
        output_module_50 = self.module_51(output_module_50)
        output_module_52 = self.module_52(dim=1, tensors=[output_module_49,output_module_50])
        output_module_53 = self.module_53(output_module_52)
        output_module_53 = self.module_54(output_module_53)
        output_module_55 = self.module_55(dim=1, tensors=[output_module_52,output_module_53])
        output_module_55 = self.module_56(output_module_55)
        output_module_57 = self.module_57(input=output_module_46, other=output_module_55, alpha=1)
        output_module_58 = self.module_58(output_module_57)
        output_module_58 = self.module_59(output_module_58)
        output_module_60 = self.module_60(dim=1, tensors=[output_module_57,output_module_58])
        output_module_61 = self.module_61(output_module_60)
        output_module_61 = self.module_62(output_module_61)
        output_module_63 = self.module_63(dim=1, tensors=[output_module_60,output_module_61])
        output_module_64 = self.module_64(output_module_63)
        output_module_64 = self.module_65(output_module_64)
        output_module_66 = self.module_66(dim=1, tensors=[output_module_63,output_module_64])
        output_module_66 = self.module_67(output_module_66)
        output_module_68 = self.module_68(input=output_module_57, other=output_module_66, alpha=1)
        output_module_69 = self.module_69(output_module_68)
        output_module_69 = self.module_70(output_module_69)
        output_module_71 = self.module_71(dim=1, tensors=[output_module_68,output_module_69])
        output_module_72 = self.module_72(output_module_71)
        output_module_72 = self.module_73(output_module_72)
        output_module_74 = self.module_74(dim=1, tensors=[output_module_71,output_module_72])
        output_module_75 = self.module_75(output_module_74)
        output_module_75 = self.module_76(output_module_75)
        output_module_77 = self.module_77(dim=1, tensors=[output_module_74,output_module_75])
        output_module_77 = self.module_78(output_module_77)
        output_module_79 = self.module_79(input=output_module_68, other=output_module_77, alpha=1)
        output_module_80 = self.module_80(output_module_79)
        output_module_80 = self.module_81(output_module_80)
        output_module_82 = self.module_82(dim=1, tensors=[output_module_79,output_module_80])
        output_module_83 = self.module_83(output_module_82)
        output_module_83 = self.module_84(output_module_83)
        output_module_85 = self.module_85(dim=1, tensors=[output_module_82,output_module_83])
        output_module_86 = self.module_86(output_module_85)
        output_module_86 = self.module_87(output_module_86)
        output_module_88 = self.module_88(dim=1, tensors=[output_module_85,output_module_86])
        output_module_88 = self.module_89(output_module_88)
        output_module_90 = self.module_90(input=output_module_79, other=output_module_88, alpha=1)
        output_module_91 = self.module_91(dim=1, tensors=[output_module_13,output_module_24,output_module_35,output_module_46,output_module_57,output_module_68,output_module_79,output_module_90])
        output_module_91 = self.module_92(output_module_91)
        output_module_91 = self.module_93(output_module_91)
        output_module_91 = self.module_94(input=output_module_91, other=output_module_0, alpha=1)
        output_module_91 = self.module_95(output_module_91)
        output_module_91 = self.module_96(output_module_91)
        output_module_91 = self.module_97(output_module_91)
        return output_module_91
