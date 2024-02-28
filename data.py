from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,txt_path,transform = None , target_transform = None):
        fh = open(txt_path,'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            img_path = words[0]
            img_path = img_path.replace('\\','/')
            img_label = int(words[1])
            imgs.append((img_path,img_label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.imgs)