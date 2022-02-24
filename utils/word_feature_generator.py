import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import torchvision.models as models

torch.cuda.set_device(0)
use_gpu = torch.cuda.is_available()
print('use_gpu=',use_gpu)
resnet101 = models.resnet101(pretrained=True)
print('resnet101=',len(list(resnet101.children())))
models = list(resnet101.children())[:-4]

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
    #print(len(out))
    return out.cuda().data.cpu().numpy()

from PIL import Image, ImageDraw, ImageFont
import csv
import os
import re
import string

save_path = "./"
img_path = './IM-MIND/'

stopwords = {'an', 'does', 'where', 'those', 'our', 'off', 'very', 'own', 'down', 'below', 'should', 'too', 'a', 'or', 'did', 'is', 'until', 'at', 'to', 'isn', 'wouldn', 'during', 'few', "mightn't", 'not', 'won', 'between', 'we', 'such', 'had', "should've", "you've", "hadn't", 'themselves', 'because', 'and', 'but', 'above', 'wasn', 'who', 'was', 'its', 'only', "shan't", 'mightn', 'couldn', 'these', "wasn't", "wouldn't", 'do', 'then', 'i', 'while', 'some', "you'll", 'having', 'more', 'he', 'mustn', "you're", "doesn't", 'me', 'll', 'weren', 'she', 'all', 'once', "hasn't", 'as', 'd', 'shouldn', 'ain', 'her', 'hers', 'they', 'aren', 'than', "isn't", 'them', 'just', 'been', 'again', 'now', 'of', 'have', 'm', 'hadn', "she's", 'haven', 'you', 'myself', 'no', 't', 'doesn', 'being', 'over', 'by', 'how', "aren't", 'my', 'doing', 'has', 'there', 'both', 'am', 'into', "it's", "you'd", 'yourself', 'can', 'ours', "haven't", 'ma', 'out', 'herself', 'each', 'o', "mustn't", 'against', 'what', 'here', 'through', 'didn', 'why', 'other', "didn't", 'yourselves', 's', 'whom', 'it', 'be', 'for', "don't", 'same', "couldn't", 'their', 'are', 're', "won't", 'your', 'up', 'theirs', 'y', "weren't", 've', 'ourselves', 'if', 'the', 'from', 'under', 'yours', 'don', 'on', 'further', "that'll", 'with', "needn't", 'nor', 'any', 'shan', 'after', 'in', 'needn', 'this', 'about', 'hasn', 'so', 'himself', 'itself', 'will', 'that', 'most', "shouldn't", 'before', 'when', 'his', 'him', 'which', 'were'}
text = {}
org = {}
word_feature_dic ={}
word_feature_map = {}
# def word_tokenize(sent):
#     pat = re.compile(r'[\w]+|[.,!?;|]')
#     if isinstance(sent, str):
#         return pat.findall(sent.lower())
#     else:
#         return []
def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def convert(str):
    maxLen = 27
    str += " "
    ans = ""
    tmp = ""
    numcnt = 0
    row = 1
    prelen = 0
    loc = 0
    for i in range (len(str)):
        numcnt = numcnt + 1
        #print(numcnt)
        if str[i] == " ":
            #word = re.sub("[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）]+", " ", tmp).strip()
            loc += 1
            #print(word,(row, loc))
            if numcnt-1 > maxLen:
                #print('i=',i)
                if row == 3:
                    if numcnt + 3 > maxLen:
                        #print(ans)
                        ans = ans[0:len(ans)-1-prelen] + "..."
                        #print(ans)
                    else :
                        ans += '...'
                    break
                row = row + 1
                loc = 0
                tmp = tmp.strip() + " "
                ans += '\n'
                ans += tmp
                numcnt = len(tmp)
                prelen = len(tmp)
                #print(numcnt)
                tmp = ""
            else:
                tmp += " "
                ans += tmp
                prelen = len(tmp)
                tmp = ""
        else:
            tmp += str[i]
    return ans

def pool3d(out, location):# average pooling for aggregating the word feature
                          # 512x28x28 with ([0,9) [9,18)) & 1024x14x14 with [0,4) [4,8) [8,)
    ans = []
    row, col, colLength, flag = location
    out = np.array(out)
    average_len = 28 // colLength
    col += 1
    if row == 0:
        if flag:
            #region = out[:, 0:4, (col-1)*average_len :]
            region = out[:, 0:9, (col - 1) * average_len:]
        else:
            #region = out[:, 0:4, (col-1)*average_len : col * average_len]
            region = out[:, 0:9, (col - 1) * average_len: col * average_len]
    elif row == 1:
        if flag:
            #region = out[:, 4:8, (col-1)*average_len :]
            region = out[:, 9:18, (col - 1) * average_len:]
        else:
            #region = out[:, 4:8, (col-1)*average_len : col * average_len]
            region = out[:, 9:18, (col - 1) * average_len: col * average_len]
    else:
        if flag:
            #region = out[:, 8:, (col-1)*average_len :]
            region = out[:, 18:, (col - 1) * average_len:]
        else:
            #region = out[:, 8:, (col-1)*average_len : col * average_len]
            region = out[:, 18:, (col - 1) * average_len: col * average_len]

    #print('col=',col,'colLength=',colLength,'region.shape=',region.shape,'averagelen=',average_len)
    for i in range(512):
        x, y = region[i].shape
        ans.append(np.sum(region[i])/x*y)
    ans = np.array(ans)
    #print('feature.shape',ans.shape)
    return ans



def split_and_extract(path, name, str):
    img = Image.open(path+name+'.jpg')
    img_title = img.crop((220, 10, 610, 140))
    #img_title.show()
    out = extractor_feature(img_title)
    #print('out.shape=',out.shape)

    str = str.replace('-', ' ')
    ans = list(filter(None, re.split(r'[\n]', convert(str))))
    #print('ans=',ans)
    anslen = len(ans)
    for i in range(anslen):
        #tmp = list(filter(None, re.split(r'[\ ]', ans[i])))
        #print(ans[i])
        tmp = re.findall(r'\b\w+\b',ans[i].lower())

        #print('tmp=',tmp)
        tmplen = len(tmp)
        for j in range(tmplen):
            #word = re.sub("[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）]+", " ", tmp[j]).strip()
            word = tmp[j].strip()
            word = word.lower()
            if j == tmplen-1:
                flag = True
            else:
                flag = False
            #print(word, (i, j, tmplen, flag))
            if word not in word_feature_dic:
                word_feature_dic[word] = [pool3d(out, (i, j, tmplen, flag))]
            else:
                word_feature_dic[word].append(pool3d(out, (i, j, tmplen, flag)))
    #print('dict=',word_feature_dic)



if __name__ == '__main__':

    for address in ["./MINDlarge_train/news.tsv", "./MINDsmall_train/news.tsv","./MINDlarge_test/news.tsv"]:
        with open(address, 'r', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter='\t')
            tsvreader_list = []
            for line in tsvreader:
                tsvreader_list.append(line)
            #length = len(list(tsvreader))
            for line in tqdm(tsvreader_list):
                newsId = line[0]
                title = line[3]
                _org = line[1]
                if _org == 'foodanddrink':
                    _org = 'food and drink'
                text[newsId] = title
                org[newsId] = _org
                split_and_extract(img_path, newsId, title)


    for k, v in tqdm(word_feature_dic.items()):
        v = np.array(v)
        word_feature_map[k] = np.sum(v,axis=0)/len(v)
    np.save('./word_feature_map_512.npy',word_feature_map)


