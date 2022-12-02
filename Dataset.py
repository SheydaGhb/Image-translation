"""
Here I define Dataset class, by this class we can iterate through input data
and target data. output of this class is pair of (input, target) that can be
fed to the network
"""

from torch.utils.data import Dataset
import numpy
import glob
from torch.utils.data import DataLoader
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, img_dir ):
        super().__init__()
        self.img_path = img_dir
        self.train_set =[]
        self.input_img = []
        self.target_img = []

        images_name = glob.glob(self.img_path + '/*_input.png') #reading the file names in the folder of images
        targets_name = glob.glob(self.img_path + '/*_target.png')

        for img_name in images_name:
            img = numpy.asarray(Image.open(img_name))  #imread input images
            self.input_img.append(img)

        for tar_name in targets_name:
            tar = numpy.asarray(Image.open(tar_name)) # imread target images
            self.target_img.append(tar)

        return

    def __len__(self):
        return len(self.input_img)

    def __getitem__(self,item):
        img = self.input_img[item]
        targ = self.target_img[item]
        return (img, targ)





"""
if you run this script you see one example of reading dataset from dataset class an d Dataloader
"""
def main():
    root_path = 'C:/Users/sheyd/OneDrive/Desktop/Visidon/visidon'  ## path to the dataset
    training_data = DataLoader(dataset=MyDataset(root_path ), batch_size=1, shuffle=False)  # loading data from Dataset class

    for train_data in training_data:
        x_train_batch, y_train_batch = train_data  ##iterating through iput image and target image of train set
        print(x_train_batch)

if __name__ == '__main__':
    main()