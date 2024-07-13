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
#from torchvision.models.resnet import resnet18
from models import RDN
import numpy as np
import cv2

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="/path/to/imagenet/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
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

def load_data(train=True,
              data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='resnet18',
              **kwargs):

  #prepare data
  # random.seed(12345)
  traindir = data_dir + '/train'
  valdir = data_dir + '/train'
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if model_name == 'inception_v3':
    size = 299
    resize = 299
  else:
    size = 224
    resize = 256
  if train:
  
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
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    dataset_opt = {'mode': 'LRHR', 
	   'dataroot_HR': valdir + '/HR_x2/',
	   'dataroot_LR': valdir + '/LR_x2/',
	   'data_type':'img',
	   'scale':2,
           }
        
    dataset = create_dataset(dataset_opt)    
    print(len(dataset))
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

def calc_psnr(img1, img2):
    print('img1')
    print(img1.shape)
    print(img2.shape)
    
    psnr = []
    for j in range(img1.shape[0]):
    	#psnr.append(10. * torch.log10(1. / torch.mean((img1[j,...] - img2[j, ...]) ** 2)))
    	psnr.append(10. * torch.log10(255.*255. / torch.mean((img1[j,...] - img2[j, ...]) ** 2)))
    
    return sum(psnr)


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    total_ssim = 0
    #print('img1shape')
    #print(img1.shape)
    for i in range(img1.shape[0]):
     
      imgc1 = img1[i, ...].detach().numpy().astype(np.float64)
      imgc2 = img2[i, ...].detach().numpy().astype(np.float64)
      kernel = cv2.getGaussianKernel(11, 1.5)
      window = np.outer(kernel, kernel.transpose())
    
      #print(img1.shape)
      #print(img2.shape)

      mu1 = cv2.filter2D(imgc1, -1, window)[5:-5, 5:-5]  # valid
      mu2 = cv2.filter2D(imgc2, -1, window)[5:-5, 5:-5]
      mu1_sq = mu1**2
      mu2_sq = mu2**2
      mu1_mu2 = mu1 * mu2
      sigma1_sq = cv2.filter2D(imgc1**2, -1, window)[5:-5, 5:-5] - mu1_sq
      sigma2_sq = cv2.filter2D(imgc2**2, -1, window)[5:-5, 5:-5] - mu2_sq
      sigma12 = cv2.filter2D(imgc1 * imgc2, -1, window)[5:-5, 5:-5] - mu1_mu2

      ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
                                                            
      total_ssim += ssim_map.mean()                                                   
                                                            
    total_ssim = total_ssim/img1.shape[0]
    return total_ssim


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    rssim = []
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 4:
       #processImage1= img1.squeeze(1)
       #processImage2= img2.squeeze(1)
       #print("processImage")
       #print(processImage1.shape)
       #print("img1")
       #print(img1.shape)
	
       for i in range(img1.shape[0]):
          vssim = ssim(img1[i,...], img2[i, ...])
          rssim.append(vssim)
    else:
        raise ValueError('Wrong input image dimensions.')
    
    return sum(rssim)


def calc_metrics(img1, img2):
    #ycrcb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    #ycrcb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)


    psnr = calc_psnr(img1, img2)/img1.shape[1]
    ssim = calc_ssim(img1, img2)
    return psnr, ssim    
    
    
   


def accuracy(output, target):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""

  #print(type(SRNP))
  #print(type(HRNP))
  #imageio.imwrite('./result.bmp', SRNP)
  #imageio.imwrite('./Com.bmp', HRNP)
  batch_size = target.shape[0]
	
  psnr, ssim = calc_metrics(output, target)
	
  print(target.shape[0])
  print(psnr)
  print(ssim)

  psnr_mean = psnr/batch_size
  ssim_mean = ssim/batch_size

  return psnr_mean, ssim_mean

# Evalution function should be called in quantization test stage. 
def evaluate(model, val_loader, loss_fn):

  model.eval()
  model = model.to(device)
  #top1 = AverageMeter('Acc@1', ':6.2f')
  #top5 = AverageMeter('Acc@5', ':6.2f')
  
  total_psnr = []
  total_ssim = []
  
  total = 0
  Loss = 0
  for iteraction, (lrimage, hrimage) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    print(lrimage.shape)
    lrimage = lrimage.to(device)
    hrimage = hrimage.to(device)
    
    
    
    #lrimage = lrimage.unsqueeze(1)
    #hrimage = hrimage.unsqueeze(1)
    #pdb.set_trace()
    #outputs = model(lrimage).clamp(0.0, 1.0)
    outputs = model(lrimage)
    
    #print(outputs.shape)
    
    for imgn in range(outputs.shape[0]):
      print(hrimage[imgn].shape)
      #origin_image = hrimage[imgn].detach().numpy().squeeze(0)
      origin_image = hrimage[imgn].detach().numpy()
      print(outputs[imgn].shape)
      #process_Image = outputs[imgn].detach().numpy().squeeze(0)
      process_Image = outputs[imgn].detach().numpy()
      
      #origin_image_name = 'o' + str(iteraction)+str(imgn)+'.bmp'
      #process_image_name = 'p' + str(iteraction)+str(imgn)+'.bmp'
      
      #origin_image= np.clip(origin_image, 0, 255)*255
      #process_Image=np.clip(process_Image, 0, 255)*255
      
      #cv2.imwrite(origin_image_name, origin_image.astype(np.uint8).transpose([ 1, 2, 0]))
      #cv2.imwrite(process_image_name, process_Image.astype(np.uint8).transpose([ 1, 2, 0]))
    
    loss = loss_fn(outputs, hrimage)
    Loss += loss.item()
    total += lrimage.size(0)
    psnr, ssim =accuracy(outputs, hrimage)
    total_psnr.append(psnr)
    total_ssim.append(ssim)
    
    print("total_psnr is:", total_psnr)
    print("total_ssim is:", total_ssim)
  return sum(total_psnr)/len(total_psnr), sum(total_ssim)/len(total_ssim), Loss / total
  
# Extracted from the upper function 'evaluate'.
# In calibration, cannot evaluate the model accuracy because the quantization scales of tensors are kept being tuned.
def forward_loop(model, val_loader):
  model.eval()
  model = model.to(device)
  #for iteraction, (images, _) in tqdm(
      #enumerate(val_loader), total=len(val_loader)):
  #  images = images.to(device)
  #  outputs = model(images)
  for iteraction, (lrimage, hrimage) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
     print(lrimage.shape)
     lrimage = lrimage.to(device)
     hrimage = hrimage.to(device)
     
     #lrimage = lrimage.unsqueeze(1)
     #hrimage = hrimage.unsqueeze(1)
     #outputs = model(lrimage).clamp(0.0, 1.0)
     outputs = model(lrimage)

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


  model = RDN(scale_factor=2,
  	      num_channels=3,
  	      num_features=32,
  	      growth_rate=32,
  	      num_blocks=8,
  	      num_layers=3
  	      ).to(device)

  #state_dict = model.state_dict()
  #parameters = torch.load(file_path, map_location=lambda storage, loc: storage)
    #model.load_state_dict(parameters['state_dict'])

  #for name in parameters['state_dict']:
  #    if name in state_dict.keys():
  #        state_dict[name].copy_(parameters['state_dict'][name])
          #print(name)
  #    else:
  #        raise KeyError(name)
  
  state_dict = model.state_dict()
  for n, p in torch.load(file_path, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

  input = torch.randn([batch_size, 3, 512, 640])
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
    print(config_file)
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  #loss_fn = torch.nn.CrossEntropyLoss().to(device)
  loss_fn = torch.nn.MSELoss().to(device)

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
          subset_len=2,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        print('fastfinetune')
        quantizer.fast_finetune(forward_loop, (quant_model, ft_loader))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  if quant_mode == 'calib':
    print(batch_size)
    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it.
    forward_loop(quant_model, val_loader)
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  else:
    psnr_gen, ssim_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
    # logging accuracy
    print('loss: %g' % (loss_gen))
    print('PSNR is: %g, SSIM is: %g, Loss is: %g' % (psnr_gen, ssim_gen, loss_gen))

  # handle quantization result
  if quant_mode == 'test' and  deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = 'best'
  file_path = os.path.join(args.model_dir, model_name + '.pth')

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
