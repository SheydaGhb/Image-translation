import numpy
from PIL import Image
from Architecture import UNet
import torch
from torch import  cuda,float32, tensor,reshape
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'

model = UNet(3,3)  ## creating one instance of the model
model.load_state_dict(torch.load("C:/Users/sheyd/OneDrive/Desktop/Visidon/mymodel_version3.pth")) ## loading the model from its path
model.to(device)  ## move it on gpu
model.eval()  ## model should be in testing phase


testimg_root = 'C:/Users/sheyd/OneDrive/Desktop/Visidon/image_for_final_testing/'  ## test image path
test_img = tensor(numpy.asarray(Image.open(testimg_root + '4764_input.png')))
test_img = test_img.unsqueeze(0).to(float32).to(device) / 255  ## prepare data for model ( change in data shape and type)
test_img = reshape(test_img, (test_img.shape[0], test_img.shape[3], test_img.shape[1], test_img.shape[2]))
y_predicted = model(test_img)
y_predicted = y_predicted.cpu().detach().numpy()  ## convert output from torch tensor to numpy
final_res = y_predicted.squeeze(0).reshape(y_predicted.shape[2], y_predicted.shape[3], y_predicted.shape[1]) ## prepare for imshow

"calculating L2Norm between models output and target images"
test_tar_py = numpy.asarray(Image.open(testimg_root + '/4764_target.png'))/255
l2 = numpy.sum(numpy.power((test_tar_py-final_res),2))
print(l2/(test_tar_py.shape[0]*test_tar_py.shape[1]))



#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,2)

axarr[0].imshow(final_res)
axarr[1].imshow(test_tar_py)
axarr[0].set_title('Output of the model')
axarr[1].set_title('Target image')
plt.show()








