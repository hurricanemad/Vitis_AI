import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
import collections
#from torchvision.models.resnet import resnet18
from srfbn import SRFBN
from srfbn import define_SRFBN
import numpy as np
import math
import cv2
import imageio

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="/path/to/medicalimage/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help=''
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()

    if mode == 'LRHR':
        from LRHR_dataset import LRHRDataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)
    dataset = D(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset
    
def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def load_data(train=True,
              data_dir='/workspace/medicalx2',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='srfbn',
              **kwargs):

  #prepare data
  # random.seed(12345)
  traindir = data_dir + '/train'
  valdir = data_dir + '/val'
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.42913573, 0.44984126, 0.63257891], std=[1.0, 1.0, 1.0])

  if train:
    #dataset_opt["mode"] = "LRHR"
    #dataset_opt["dataroot_HR"] = traindir + 'HR_x2'
    #dataset_opt["dataroot_LR"] = traindir + 'LR_x2'
    #dataset_opt["rgb_range"] = 255
    #dataset_opt["phase"] = "train"
    #dataset_opt["scale"] = 2
    #dataset_opt["data_type"] = "img"
    #dataset_opt["LR_size"] = "40"
    #dataset_opt["noise"] = "."
    
    dataset_opt = {'mode': 'LRHR', 
    		   'dataroot_HR': valdir + '/HR_x2/',
    		   'dataroot_LR': valdir + '/LR_x2/',
    		   'phase': 'test',
    		   'scale':2,
    		   'data_type':'img',
                   }
    
    
    dataset = create_dataset(dataset_opt)
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    #dataset_opt["mode"] = "LRHR"
    #dataset_opt["dataroot_HR"] = valdir + 'HR_x2'
    #dataset_opt["dataroot_LR"] = valdir + 'LR_x2'
    #dataset_opt["rgb_range"] = 255
    #dataset_opt["phase"] = "val"
    #dataset_opt["scale"] = 2
    #dataset_opt["data_type"] = "img"
    
    dataset_opt = {'mode': 'LRHR', 
	   'dataroot_HR': valdir + '/HR_x2/',
	   'dataroot_LR': valdir + '/LR_x2/',
	   'rgb_range':255,
	   'phase': 'test',
	   'scale':2,
	   'data_type':'img',
	   'LR_size': 40,
	   'noise': '.'
           }
        
    dataset = create_dataset(dataset_opt)        
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))    
 
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type) 
    
def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()    
    
def Tensor2np(tensor_list, rgb_range):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]
    
def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim    

def accuracy(output, target, scale):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    SRImage = output.data[0].float().cpu()
    HRImage = target.data[0].float().cpu()
    SRNP = Tensor2np([SRImage], 255)[0]
    HRNP = Tensor2np([HRImage], 255)[0]
    #print(type(SRNP))
    #print(type(HRNP))
    #imageio.imwrite('./result.bmp', SRNP)
    #imageio.imwrite('./Com.bmp', HRNP)

    batch_size = target.size(0)
	
    psnr, ssim = calc_metrics(SRNP, HRNP, scale)

    psnr_mean = psnr/batch_size
    ssim_mean = ssim/batch_size

    return psnr_mean, ssim_mean

# Evalution function should be called in quantization test stage. 
def evaluate(model, val_loader, loss_fn):

  model.eval()
  model = model.to(device)
  #psnr_eval = AverageMeter('psnr', ':6.2f')
  #ssim_eval = AverageMeter('ssim', ':6.2f')
  total_psnr = []
  total_ssim = []
  
  total = 0
  Loss = 0
  for iteraction, (lrimage, hrimage) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
       
    lrimg = torch.FloatTensor()
    hrimg = torch.FloatTensor()
    
    lrimg.resize_(lrimage.size()).copy_(lrimage)
    hrimg.resize_(hrimage.size()).copy_(hrimage)
    
    print(lrimage.size())
    print(hrimage.size())
    #lrimage = lrimage.to(device)
    #hrimage = hrimage.to(device)
    ##pdb.set_trace()
    #out_hrimage = model(lrimage)
    with torch.no_grad():
       out_hrimage = model.forward(lrimg)
       if isinstance(out_hrimage, list):
          out_hrimage = out_hrimage[-1]
       else:
          out_hrimage = out_hrimage
    
    #print(out_hrimage.size(0))   
    #print(hrimg.size(0))   
    loss = loss_fn(out_hrimage, hrimg)
    Loss += loss.item()
    total += lrimage.size(0)
    psnr, ssim = accuracy(out_hrimage, hrimg, 2)
    #psnr_eval.update(psnr, lrimage.size(0))
    #ssim_eval.update(ssim, lrimage.size(0))
    total_psnr.append(psnr)
    total_ssim.append(ssim)

  return sum(total_psnr)/len(total_psnr), sum(total_ssim)/len(total_ssim), Loss / total
  
# Extracted from the upper function 'evaluate'.
# In calibration, cannot evaluate the model accuracy because the quantization scales of tensors are kept being tuned.
def forward_loop(model, val_loader):
  model.eval()
  model = model.to(device)
  for iteraction, (images, _) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    images = images.to(device)
    outputs = model(images)

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  #model = resnet18().cpu()
  
  #opt['in_channels'] = 3
  #opt['out_channels'] = 3
  #opt['num_features'] = 64
  #opt['num_steps'] = 4
  #opt['num_groups'] = 6
  #opt['scale'] = 2
  opt = {'in_channels':3, 'out_channels':3, 'num_features':32, 'num_steps':4, 'num_groups':3, 'scale':2}
  
  model = define_SRFBN(opt).cpu()
  #model.load_state_dict(torch.load(file_path,map_location='cpu'))
  #checkpoint = torch.load(file_path,map_location='cpu')
  #if 'state_dict' in checkpoint.keys(): 
      #checkpoint = checkpoint['state_dict']
  #load_func = model.load_state_dict
  #load_func(checkpoint)
  
  #print(isinstance(model, torch.nn.DataParallel))
  
  #checkpoint=torch.load(file_path,map_location='cpu')
  #new_pth=model.state_dict() # 需要加载参数的模型
    # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in new_pth}
  #pretrained_dict={} # 用于保存公共具有的参数
  #for k,v in checkpoint['state_dict'].items():
     #for kk in new_pth.keys():
         #if kk in k:
             #pretrained_dict[kk]=v
             #break
  #new_pth.update(pretrained_dict)
  #model.load_state_dict(new_pth)
  
  state_dict = torch.load(file_path, map_location='cpu')

  # 创建一个新的OrderedDict，移除`module.`前缀
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k, v in state_dict['state_dict'].items():
    name = k[7:]  # 移除`module.`前缀
    new_state_dict[name] = v

  # 加载模型
  model.load_state_dict(new_state_dict)

  #print(batch_size)

  input = torch.randn([batch_size, 3, 224, 224])
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input,), device=device)
      sys.exit()
      
  else:
    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  #loss_fn = torch.nn.CrossEntropyLoss().to(device)
  
  loss_fn = torch.nn.L1Loss().to(device)

  val_loader, _ = load_data(
      subset_len=subset_len,
      train=False,
      batch_size=batch_size,
      sample_method='random',
      data_dir=data_dir,
      model_name=model_name)

  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=5120,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(forward_loop, (quant_model, ft_loader))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  if quant_mode == 'calib':
    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it.
    forward_loop(quant_model, val_loader)
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  else:
    psnr_gen, ssim_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
    # logging accuracy
    print('loss: %g' % (loss_gen))
    print('psnr is: %g ssim is: %g ' % (psnr_gen, ssim_gen))

  # handle quantization result
  if quant_mode == 'test' and  deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = 'SRFBN'
  file_path = os.path.join(args.model_dir, model_name + '.pth')
  print(file_path)
  
  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))
