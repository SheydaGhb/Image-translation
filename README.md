# Image translation
## Translation of an image to a target image.


I designed a model in Pytorch that performs image translation from the input images to the target images. An example of the input image and its target image is shown below

<img width="459" alt="image" src="https://user-images.githubusercontent.com/31028574/205402113-856844ba-d393-46a2-82a4-6f7ad8c6bf9e.png">

## For training : 
Run  ``` training.py ``` script. <br /> 
Dependencies are : <br />
Python 3.10 <br />
Numpy 1.23.5 <br />
Torch 1.12.1 <br />
Torchvison 0.13.1 <br />
Cuda 11.6 <br />
Matplotlib 3.6.2 <br />
Pillow 9.3.0 <br />

## For testing a single image 
Download the model from [here](https://tuni-my.sharepoint.com/:u:/g/personal/sheyda_ghanbaralizadehbahnemiri_tuni_fi/EefhTnBnXmlPgWGjU9seFfkBArrboa-Zocw9v7xqPnRsAQ?e=WNf0AO). Model's name is mymodel_version3 <br />
Run  ``` test_oneimage.py ``` on a test image<br />

Dataset for training anf testing is provided by [Visidon](https://www.visidon.fi/)
## Results
![image](https://user-images.githubusercontent.com/31028574/205401871-ad4169c6-cdc8-4712-8a68-a540026e01f9.png)
![image](https://user-images.githubusercontent.com/31028574/205401882-bc9531eb-bf7c-4512-b5ef-96c9294ede55.png)







