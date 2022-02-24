import numpy as np
import torch
import torch.nn as nn
import os
import torchvision.models as models
from tqdm import tqdm
resnet101 = models.resnet101(pretrained=True)

#torch.cuda.set_device(2)
resnet101 = models.resnet101(pretrained=True)
models = list(resnet101.children())[:-1]
resnet101 = nn.Sequential(*models)
resnet101 = resnet101.cuda()
for p in resnet101.parameters():
    p.requires_grad = False

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

from PIL import Image

from torchvision import transforms
transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
)])

def extractor_feature(img):
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.cuda()
    resnet101.eval()
    out = resnet101(batch_t)
    out = out.squeeze()
    #print(out.shape)
    # print(len(out))
    return out.cuda().data.cpu().numpy()


if __name__ == '__main__':
    #print(list(models.resnet101(pretrained=True).children()))
    path = './IM-MIND/'
    save_path = './'
    #cnt = 0
    feature_global = {}
    for filename in tqdm(os.listdir(path)):
        _filename = filename.strip('.jpg')
        #cnt = cnt + 1
        #print(cnt,' ',filename)
        img = Image.open(path+filename)
        feature_global[_filename] = extractor_feature(img)
    np.save(save_path + 'feature_global.npy', feature_global)
