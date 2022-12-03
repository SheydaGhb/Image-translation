from torch import  cuda,float32,Tensor,reshape,no_grad
import torch
from torch.utils.data import DataLoader,random_split
from Dataset import MyDataset
from Architecture import UNet
from torch.optim import Adam
from torch.nn import MSELoss
from pathlib import Path
import time

def main():
    print("Running on Cuda:" , torch.cuda.is_available())  ## see if code is running on cuda
    epochs = 60
    device = 'cuda' if cuda.is_available() else 'cpu'

    root_path = 'C:/Users/sheyd/OneDrive/Desktop/Visidon/VD_dataset2'  ## path to the trainig data, modify it according to your system's path


    full_dataset = MyDataset(root_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(MyDataset(root_path), [train_size, test_size])  ##spliting data randomly to 80% and 20%
    training_data = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)
    testing_data = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)


    mymodel = UNet(3,3) ## instance of the model
    mymodel = mymodel.to(device) ## move the model to run on gpu

    loss_function = MSELoss()
    optim = Adam(params=mymodel.parameters(), lr=1e-3)  ## activate Adam optimizer

    ## start training
    print(' start training...')
    start_time = time.time()
    for epoch in range(epochs):

        epoch_loss_training = []
        epoch_loss_test=[]
        mymodel.to(device)
        mymodel.train()


        for train_data in training_data:
            optim.zero_grad()
            x_train_batch, y_train_batch = train_data
            ## data should be reshaped from (N,H,W,C) to (N,C,H,W)
            x_train_batch = reshape(x_train_batch,(x_train_batch.shape[0], x_train_batch.shape[3], x_train_batch.shape[1],x_train_batch.shape[2]))
            y_train_batch = reshape(y_train_batch,(y_train_batch.shape[0], y_train_batch.shape[3], y_train_batch.shape[1],y_train_batch.shape[2]))
            x_train_batch = x_train_batch.to(float32).to(device)/255
            y_train_batch = y_train_batch.to(float32).to(device)/255


            y_predict = mymodel(x_train_batch)  ## calling mymodel to process the train data and create an output
            loss = loss_function(y_predict, y_train_batch) # calculating MSEloss

            loss.backward()
            optim.step()
            epoch_loss_training.append(loss.item())
            loss_training_mean = Tensor(epoch_loss_training).mean()

        ## evaluation process
        mymodel.eval()
        with no_grad():
            for test_data in testing_data:
                x_test_batch, y_test_batch = test_data
                x_test_batch = x_test_batch.to(float32).to(device)/255
                y_test_batch = y_test_batch.to(float32).to(device)/255
                x_test_batch = reshape(x_test_batch, (x_test_batch.shape[0], x_test_batch.shape[3], x_test_batch.shape[1], x_test_batch.shape[2]))
                y_test_batch = reshape(y_test_batch, (y_test_batch.shape[0], y_test_batch.shape[3], y_test_batch.shape[1], x_test_batch.shape[2]))
                y_predicted = mymodel(x_test_batch)

                eval_loss = loss_function(y_predicted, y_test_batch)
                epoch_loss_test.append(eval_loss.item())
                loss_test_mean = Tensor(epoch_loss_test).mean()


            print(f'Epoch: {epoch:03d} | 'f'Training loss: {loss_training_mean:7.4f} 'f'validation loss: {loss_test_mean:7.4f} ')

    ##save the model in your system, path should be changed accordingly.
    torch.save(mymodel.state_dict(), Path('C:/Users/sheyd/OneDrive/Desktop/Visidon/mymodel_version3.pth'))

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
