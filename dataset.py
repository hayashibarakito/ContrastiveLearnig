import os
import glob
import torch
from PIL import Image
import numpy as np

from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from torch.utils.data import Dataset,DataLoader

def make_datapath_dic(phase='train'):
    root_path = './flickr/' + phase
    class_list = os.listdir(root_path)
    class_list = [class_name for class_name in class_list if not class_name.startswith('.')]
    datapath_dic = {}
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name, '*.jpg')
        for path in glob.glob(target_path):
            data_list.append(path)
        datapath_dic[i] = data_list

    return datapath_dic

class ImageTransform():
    def __init__(self, size, s=1):
            """Return a set of data augmentation transformations as described in the SimCLR paper."""
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            self.data_transform  = {'train':    transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor()
                                    ]),
                                    'test':     transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                                    ])
                                    }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class SupConDataset(torch.utils.data.Dataset):
    def __init__(self, datapath_dic, transform, phase='train'):
        self.datapath_dic = datapath_dic
        self.transform = transform
        self.phase = phase

        all_datapath = []
        bins = [0]
        for data_list in self.datapath_dic.values():
            all_datapath += data_list
            bins.append(bins[-1] + len(data_list))
        self.all_datapath = all_datapath
        self.bins = bins

    def __len__(self):
        return len(self.all_datapath)

    def __getitem__(self, idx):
        path = self.all_datapath[idx]
        for i in range(len(self.bins)):
            if idx < self.bins[i]:
                label = i
                break
        img = Image.open(path)
        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/util.py#L9
        img_1 = self.transform(img, self.phase)
        img_2 = self.transform(img, self.phase)
        img = [img_1, img_2]
        return {"image": img, "target": label}

"""
dic = make_datapath_dic("train")
transform = ImageTransform(300)
train_dataset = SupConDataset(dic, transform=transform, phase="train")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#写真確認
import matplotlib.pyplot as plt
def image_show(train_loader,n):

  #Augmentationした画像データを読み込む
  tmp = iter(train_loader)
  mydict = tmp.next()

  #画像をtensorからnumpyに変換
  images1 = mydict["image"][0].numpy()
  images2 = mydict["image"][1].numpy()
  #print(mydict["target"])

  #n枚の画像を1枚ずつ取り出し、表示する
  for i in range(n):
    image1 = np.transpose(images1[i],[1,2,0])
    plt.imshow(image1)
    plt.show()
    image2 = np.transpose(images2[i],[1,2,0])
    plt.imshow(image2)
    plt.show()

image_show(train_loader,10)
"""