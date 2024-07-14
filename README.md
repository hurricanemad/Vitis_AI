# Vitis_AI
## Introduction:

**This is a modified Vitis-AI cloned from Xilinx repository(https://github.com/Xilinx/Vitis-AI). These codes run in a docker environment created by Xilinx. You can browse the [quick start guide](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html) of Vitis-AI for Zynq UltraScale+ to learn about the methods to intialize the docker environment. The program provide the quantizing code of the model of two super resolution algorithm, which are [RDN](Residual Dense Network for Image Super-Resolution)  and [SRCNN](Super-Resolution Convolutional Neural Network). The quantized model can finally run in Kria KR260 Robotic kit.**

## Author: Hui Shen(Doxxxx)

## Date: May, 2024

## Quick Start:

### Premise: 

You should train a float model of the [RDN](https://github.com/yjn870/RDN-pytorch) or [SRCNN](https://github.com/yjn870/SRCNN-pytorch) network using the goal image dataset.

### Example:

**You can use below code to quantizing a RDN model:**

**sudo su**

**cd repository Path**

You can initialize the docker environment of pytorch cpu using below code,
the operations are likely to take a few minutes:

**./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest**

**cd /workspace/examples/vai_quantizer/pytorch/RDN**

if you would like to know the PSNR and SSIM of the intialized float model using below code,
I had trained a model using the [Kvasir subset](https://paperswithcode.com/dataset/kvasir) :

**python RDN_quant.py --quant_mode float --data_dir /workspace/medicaltempx2df --model_dir model --batch_size 1**

You can use below code to be aware of whether the model is suitable the goal architecture KR260:

**python RDN_quant.py --quant_mode float --inspect --target DPUCZDX8G_ISA1_B4096 --model_dir model --batch_size 1**

After comfirming the model can delpy in the KR260, you can use below code to start the quantization of the float model,
these operations would take hours, because we choose the cpu mode to quantize the model:

**python RDN_quant.py --quant_mode calib --data_dir /workspace/MedicalImagePatch1 --model_dir /workspace/examples/vai_quantizer/pytorch/RDN/model/  --subset_len 200 --batch_size 2**

You can evaluate the PSNR and SSIM of the quantized mode using the below code,
You can review the differences between the PSNR and SSIM befor and after model quantizing:

**python RDN_quant.py --model_dir /workspace/examples/vai_quantizer/pytorch/RDN/model/ --data_dir /workspace/medicaltempx2df --quant_mode test --batch_size 1**

You can generate the .xmodel file to run in the KR260 kit using the below code:

**python RDN_quant.py--quant_mode test --subset_len 1 --batch_size=1 --model_dir /workspace/examples/vai_quantizer/pytorch/RDN/model --data_dir /workspace/medicalx2df --deploy**

Finaly, you can use below code to convert the quantized model to the goal architecture:

**vai_c_xir -x /workspace/examples/vai_quantizer/pytorch/RDN/quantize_result/RDN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o RDN_pt -n RDN_pt**

Then you can find the quantized model in the /workspace/examples/vai_quantizer/pytorch/RDN/RDN_pt path.

After the quantization, you can transmit the model to the KR260 kit using the scp command and using the [Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI?tab=readme-ov-file) to execute the super-resolution operations.

If you want to try to quantizing the SRCNN model, you can change the above commands and model path to **SRCNN_quant.py** and **SRCNN/model**, respectively.

