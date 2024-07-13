# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class SRCNN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SRCNN::input_0(SRCNN::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[9, 9], stride=[1, 1], padding=[4, 4], dilation=[1, 1], groups=1, bias=True) #SRCNN::SRCNN/Conv2d[conv1]/ret.3(SRCNN::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SRCNN::SRCNN/ReLU[relu]/235(SRCNN::nndct_relu_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #SRCNN::SRCNN/Conv2d[conv2]/ret.5(SRCNN::nndct_conv2d_3)
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #SRCNN::SRCNN/ReLU[relu]/256(SRCNN::nndct_relu_4)
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #SRCNN::SRCNN/Conv2d[conv3]/ret(SRCNN::nndct_conv2d_5)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        return output_module_0
