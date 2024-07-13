import torch.utils.data as data
import os
import numpy as np

import cv2 as cv

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images   

def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR = get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr= self._load_file(idx)


        return lr, hr


    def __len__(self):
        #if self.train:
            #return len(self.paths_HR) * self.repeat
        #else:
        return len(self.paths_LR)


    def _load_file(self, idx):
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        
        
        #lr = common.read_img(lr_path, self.opt['data_type'])
        #hr = common.read_img(hr_path, self.opt['data_type'])

        lr_img = cv.imread(lr_path, cv.IMREAD_COLOR)
        hr_img = cv.imread(hr_path, cv.IMREAD_COLOR)
	
        #resz_lr_img = cv.resize(lr_img,(lr_img.shape[1]*2,lr_img.shape[0]*2),interpolation=cv.INTER_CUBIC)
	
        #lr_resize = np.expand_dims(resz_lr_img.astype(np.float32).transpose([2, 0, 1]), 0)/255.0
        #hr = np.expand_dims(hr_img.astype(np.float32).transpose([2, 0, 1]), 0)/255.0
        #lr_resize = resz_lr_img.astype(np.float32).transpose([2, 0, 1])/255.0
        lr_resize = lr_img.astype(np.float32).transpose([2, 0, 1])/255.0
        hr = hr_img.astype(np.float32).transpose([2, 0, 1])/255.0



        return lr_resize, hr




