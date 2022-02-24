import os
import csv
from PIL import Image, ImageDraw, ImageFont

def convert(str): #transform the whole sentences into the proper format for news card
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
        if str[i] == " ":
            #word = re.sub("[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）]+", " ", tmp).strip()
            loc += 1
            if numcnt-1 > maxLen:
                if row == 3:
                    if numcnt + 3 > maxLen:
                        ans = ans[0:len(ans)-1-prelen] + "..."
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

text = {}
org = {}
saved_img = []
save_path = '' #the path for result
img_path = '' #the path for cover images which can be crawled by yourself
news_path = '' #the path for news.tsv which is in the original dataset
for filename in os.listdir(img_path):
    saved_img.append(filename)

cnt = 0
with open(news_path, 'r', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        filename = line[0]+'.jpg'
        if filename in saved_img:
                continue
        cnt += 1
        base_img = Image.new('RGB', (610, 195), (255, 255, 255))
        font = ImageFont.truetype('seguisb.ttf', 27)
        draw = ImageDraw.Draw(base_img)
        position = (227.725, 15)
        draw.text(position, convert(text[filename.strip('.jpg')]), font=font, fill="#2b2b2b", spacing=10.5, align='left')
        font = ImageFont.truetype('segoeui.ttf', 24)
        position = (227.725, 142.5)
        draw.text(position, org[filename.strip('.jpg')], font=font, fill="#666666", spacing=15, align='left')
        base_img.save(save_path+filename)
        #base_img.show()
print(cnt)