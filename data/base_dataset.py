import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1*random.randint(0,4)
        osize = [int(768*zoom), int(1024*zoom)]
        transform_list.append(transforms.Resize(osize, Image.NEAREST))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))  
    elif opt.resize_or_crop == 'no':
        osize = [opt.fineSize, opt.fineSize]
        transform_list.append(transforms.Resize(osize, Image.NEAREST)) #因为要对二维码进行插值，所以应该用NEAREST比较哦好
    elif opt.resize_or_crop == 'resize':
        osize = [opt.fineSize, opt.fineSize]
        transform_list.append(transforms.Resize(osize, Image.NEAREST))  #音


    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transforms(opt):
    transform_list = []
  

    if opt.transform == 'resize_and_crop':

        transform_list.append(transforms.RandomResizedCrop(opt.fineSize,scale=(0.08,1),interpolation=Image.BILINEAR))
    elif opt.transform == 'resize':
        osize = [opt.fineSize , opt.fineSize]
        transform_list.append(transforms.Resize(osize, Image.NEAREST))
    elif opt.transform == 'no':
        ...
        # do nothing
    else:
        raise NotImplementedError('transform [%s] is not found' % opt.transform)

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
