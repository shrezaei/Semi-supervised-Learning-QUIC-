import pdb
import torch
import numpy as np
import myDataset
import Model
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim import lr_scheduler
import tqdm
import torch.nn as nn
from torch import optim

# Some Global Variables
# Training Parameters
learning_rate = 0.0001
epsilon = 1e-08
weight_decay = 0
batch_size = 64

# Network Parameters
time_steps = 45
output_time_steps = 1
input_size = 24
output_size = 2 * output_time_steps    #for unsupervised learning
num_classes = 5     #for supervised learning 
statisticalFeatures = 24
hidden_dim = 10
num_epoches = 200
num_layers = 1

display_step = 10

use_gpu = True  # GPU enable

tqdm.monitor_interval = 0

total_acc_single = 0
total_acc = np.zeros(num_classes)
total_pre = np.zeros(num_classes)
total_rec = np.zeros(num_classes)
total_f1 = np.zeros(num_classes)
full_acc = []
full_pre = []
full_rec = []
full_f1 = []

NumOfCrossValidationFolds = 5
Accuracies = np.zeros(NumOfCrossValidationFolds)
for foldNumber in range(NumOfCrossValidationFolds):
    # Dataset loading
#    data_directory = "Data-incremental-45-22-1.6"
    data_directory = "Data-simple-45-22"
#    data_directory = "Data-random-45-22"
#    data_directory = "temp3"
#    data_directory = "Data-simple-75"
    super_trainData = np.load(data_directory + "/re-training_trainData-" + str(foldNumber) + ".npy")
    super_trainLabel = np.load(data_directory + "/re-training_trainLabel-" + str(foldNumber) + ".npy")-1
    testData = np.load(data_directory + "/re-training_testData-" + str(foldNumber) + ".npy")
    testLabel = np.load(data_directory + "/re-training_testLabel-" + str(foldNumber) + ".npy")-1

    #pdb.set_trace()

    print(super_trainData.shape)
    print(testData.shape)


    #pdb.set_trace()

    train_dataset = myDataset.super_Dataset(super_trainData,
                                              super_trainLabel, testData, testLabel, train=True)

    test_dataset = myDataset.super_Dataset(super_trainData,
                                             super_trainLabel, testData, testLabel, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    print("Dataset loaded!")


    encoder = Model.CNNEncoder(hidden_dim, statisticalFeatures)
    #encoder = Model.FCNN(input_size, 10, time_steps)
    
    # Load the pre-trained model here:
    encoder = torch.load('simple-45-22.pth')
#    encoder = torch.load('simple-45-10.pth')
#    encoder = torch.load('simple-75.pth')
#    encoder = torch.load('incremental-45-22-1.6.pth')
#    encoder = torch.load('random-45-22.pth')
#    encoder = torch.load('temp.pth')
#    encoder = torch.load('wit-incremental-CS5-4-8-1.2.pth')
#    encoder = torch.load('wit-simple-CS5-3-10.pth')
#    encoder = torch.load('wit-random-CS5-3-0.15.pth')

    # For transfer learning, the convolutional layers can be fixed
    # layers freeze
    ct = 0
    for child in encoder.children():
        #ct<-1: re-train all layers
        #ct<1: fix only cnnseq part, not regressor
        #ct<2: fix cnnseq and regressor
        if ct <-1:
            for param in child.parameters():
                param.requires_grad = False
        ct += 1

    #
    encoder.reggresor = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128)
    )
    encoder.output_size = 128
    
#    linear = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, num_classes) )
    linear = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True),   nn.Linear(128, num_classes), nn.Softmax())
    transferedModel = nn.Sequential(encoder, linear)



    if use_gpu:
        transferedModel = transferedModel.cuda()

    # loss and optimizer
    loss_function = torch.nn.CrossEntropyLoss()

    #When freezing layer, this only works
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,transferedModel.parameters()), lr=learning_rate,eps=epsilon, weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Decay Learning Rate by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    train_acc = []
    test_acc = []

    isFirst = 1

    for epoch in tqdm.tqdm(range(num_epoches)):
        print()
        running_loss = 0.0
        running_acc = 0.0
        avg_loss = 0.0
        avg_acc = 0.0
        total_target = 0

        #exp_lr_scheduler.step()

        for i, data in enumerate(train_loader, 1):
            seq, target = data
            if len(seq) < 2:
                continue
            seq = Variable(seq).float()
            target = Variable(target).long()

            #pdb.set_trace()

            if use_gpu:
                seq = seq.cuda()
                target = target.cuda()


            if isFirst == 1:
                sample = seq[:6]
                sampleTarget = target[:6]
                ifFirst = 0

            out = transferedModel(seq)
            loss = loss_function(out, target)

            # prediction
            _, pred = torch.max(out, 1)
            total_target += len(pred)
            num_correct = (pred == target).sum()

            running_acc += num_correct.item()
            running_loss += loss.item() * target.size(0)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #print(sample)
        result = transferedModel(sample)
        t1 = sampleTarget.data[:].cpu().numpy()
        t2 = result.data[:].cpu().numpy()
        print(np.around(t1,4))
        print(np.around(t2,4))


        if epoch % display_step == 0 or epoch==num_epoches-1:
            print(i)
            avg_loss = running_loss / (total_target)
            avg_acc = running_acc / (total_target)
    
            print('Loss: {:.6f}, Acc: {:.6f}'.format(
                        avg_loss,
                        avg_acc))
    
            train_acc.append(avg_acc)
    
            # test
            transferedModel.eval()
            eval_loss = 0.
            eval_acc = 0.
            for i, data in enumerate(test_loader, 1):
                seq, target = data
                seq = Variable(seq).float()
                target = Variable(target).long()
    
                if use_gpu:
                    seq = seq.cuda()
                    target = target.cuda()
    
                out = transferedModel(seq)
                loss = loss_function(out, target)
    
                # prediction
                eval_loss += loss.item() * target.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == target).sum()
                eval_acc += num_correct.item()
    
            avg_acc = eval_acc / (len(test_dataset))
            test_acc.append(avg_acc)
            print('Test Loss{:.1f}: {:.6f}, Acc: {:.6f}'.format(foldNumber, eval_loss / (len(
                    test_dataset)), avg_acc))

        # Save the Model after each epoch
#        torch.save(transferedModel.state_dict(), './supervised.pth')


    print("Optimization Done!")
    t = np.arange(len(train_acc))

    np.set_printoptions(precision=3)
    TP = np.zeros(num_classes)
    TN = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    
    transferedModel.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(test_loader, 1):
        seq, target = data
        seq = Variable(seq).float()
        target = Variable(target).long()

        if use_gpu:
            seq = seq.cuda()
            target = target.cuda()

        out = transferedModel(seq)
        loss = loss_function(out, target)

        # prediction
        eval_loss += loss.item() * target.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == target).sum()
        eval_acc += num_correct.item()
        
        for i in range(len(pred)):
            if target[i] == pred[i]:
                TP[target[i]] += 1
                for j in range(num_classes):
                    if j != target[i]:
                        TN[j] += 1
            elif target[i] != pred[i]:
                FP[pred[i]] += 1
                FN[target[i]] += 1
    avg_acc = eval_acc / (len(test_dataset))

    Accuracy = np.true_divide(TP+TN, TP+TN+FP+FN)
    precision = np.true_divide(TP, TP+FP)
    recall = np.true_divide(TP, TP+FN)
    F1 = 2 * np.true_divide(np.multiply(precision,recall), precision+recall)
    print("Fold ", foldNumber, ":")
    print(TP,TN)
    print(FP,FN)
    print("Accuracy: ", Accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    full_acc.extend(Accuracy)
    full_pre.extend(precision)
    full_rec.extend(recall)
    full_f1.extend(F1)
    total_acc += Accuracy
    total_pre += precision
    total_rec += recall
    total_f1 += F1
    total_acc_single += avg_acc
    Accuracies[foldNumber] = avg_acc

print("Evaluation is done!")
print("Accuracy: ", full_acc)
print("precision: ", full_pre)
print("recall: ", full_rec)
print("F1: ", full_f1)
print("--------------")
print("Accuracy: ", 100*total_acc_single/NumOfCrossValidationFolds)
print(Accuracies)
print("Accuracy: ", total_acc/NumOfCrossValidationFolds)
print("precision: ", total_pre/NumOfCrossValidationFolds)
print("recall: ", total_rec/NumOfCrossValidationFolds)
print("F1: ", total_f1/NumOfCrossValidationFolds)


