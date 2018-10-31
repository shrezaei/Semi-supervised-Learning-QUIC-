import pdb
import torch
import numpy as np
import time
import random
import myDataset
import Model
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from torch import optim

tqdm.monitor_interval = 0

# Some Global Variables
# Training Parameters
learning_rate = 0.001
batch_size = 32


Mode = 0  # 0:Encoder-Linear     1:Encoder-Decoder
# Network Parameters
time_steps = 45
#output_time_steps = 15
input_size = 2    #100byte of payload + length + direction + timestamp
output_size = 24
hidden_dim = 10     #number of features in hidden layer
num_epoches = 250
num_layers = 1    #number of hidden layer


display_step = 5

teacher_forcing_ratio = 0.5

unsuper_trainData = np.load("pretraining_trainData.npy")
unsuper_trainLabel = np.load("StatLabel.npy")

#pdb.set_trace()

sample_size = unsuper_trainData.shape[0]
perm = np.arange(sample_size)
#np.random.shuffle(perm)
#unsuper_trainData = unsuper_trainData[perm]
#unsuper_trainLabel  = unsuper_trainLabel[perm]
testStart = int(sample_size * 0.96)
testData = unsuper_trainData[testStart:]
testLabel = unsuper_trainLabel[testStart:]
unsuper_trainData = unsuper_trainData[:int(testStart/1)]
unsuper_trainLabel = unsuper_trainLabel[:int(testStart/1)]

train_dataset = myDataset.unsuper_Dataset(unsuper_trainData,
                                          unsuper_trainLabel)

test_dataset = myDataset.unsuper_Dataset(testData, testLabel)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#test_loader = torch.utils.data.DataLoader(
#    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print("Dataset loaded!")

if Mode == 0:
    use_gpu = True  # GPU enable
    
    encoder = Model.CNNEncoder(hidden_dim, output_size)
    
    if use_gpu:
        encoder = encoder.cuda()
        #decoder = decoder.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    loss_function = nn.MSELoss()
#    loss_function = nn.KLDivLoss()

    #optimizer = optim.RMSprop(autoencoder.parameters(), lr = learning_rate, weight_decay = 0.005)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)#, weight_decay = 0.005)
    train_loss = []

#    sample = torch.randn(1,2,time_steps)
#    sample = Variable(sample).float()
#    sample = sample.cuda()
#    print(sample)
    isFirst = 1
    for epoch in range(num_epoches):
        running_loss = 0.0
        avg_loss = 0.0
        for i, data in enumerate(train_loader, 1):
            seq, target = data
            #target = target[:,1,:]
            #target = target[:,1,:]
            #target = seq[:,1,15:30]
            #target = torch.ones(target.shape[0],1) / 1 * 1
            #pdb.set_trace()
            
            seq = Variable(seq).float()
            target = Variable(target).float()
    
            if use_gpu:
                seq = seq.cuda()
                target = target.cuda()

            if isFirst == 1:
                sample = seq[:6,:]
                sampleTarget = target[:6,:]
#                pdb.set_trace()
#                sample = seq[:6,:,:]
#                sampleTarget = target[:6,:]
                ifFirst = 0
                
#            pdb.set_trace()
            #print(encoder.enc_linear_1.weight.data)
            out = encoder(seq)
            loss = loss_function(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * target.size(0)
                
            #pdb.set_trace()


        if epoch % display_step == 0:
            avg_loss = running_loss / (batch_size * i)
            print('[{}/{}] Loss: {:.6f}'.format(epoch + 1, num_epoches, avg_loss))
            #pdb.set_trace()
            train_loss.append(avg_loss)
            
            eval_loss = 0.
            eval_acc = 0.

        torch.save(encoder, 'simple-45.pth')
        torch.save(encoder, 'random-45.pth')
        torch.save(encoder, 'incremental-45.pth')


print("Optimization Done!")
t = np.arange(len(train_loss))
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.plot(t, train_loss, color="red", linewidth=2.5,
         linestyle="-", label="Unsupervised Loss")
plt.legend(loc='upper right')
plt.show()

