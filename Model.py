import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch.nn import init

class CNNEncoder(nn.Module):
    
    def __init__(self, hidden_size, output_size, channels=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size


        self.cnnseq = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            #nn.Conv1d(64, 128, kernel_size=3, stride=2, bias=False),
            #nn.BatchNorm1d(128),
            #nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64), #128
        )
        self.reggresor = nn.Sequential( #30:64 #45:128, #60:
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
#            nn.Dropout(0.5),
            nn.Linear(256, self.output_size)
        )
    def forward(self, images):

        code = self.cnnseq(images)
#        pdb.set_trace()
        code = code.view([images.size(0), -1])
        code = self.reggresor(code)
        code = code.view([code.size(0), self.output_size])
        return code   



class FCNN(nn.Module):
    def __init__(self, input_size, output_size , time_steps=30):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps

        self.lin1 = nn.Linear(input_size*time_steps, input_size*time_steps)
        self.lin2 = nn.Linear(input_size*time_steps, input_size*time_steps)
        self.lin3 = nn.Linear(input_size*time_steps, input_size*time_steps)
        self.lin4 = nn.Linear(input_size*time_steps, output_size)
        self.Sigmoid = nn.Sigmoid()
        
        init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0.01)
        init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0.01)
        init.xavier_uniform_(self.lin3.weight)
        self.lin3.bias.data.fill_(0.01)
        init.xavier_uniform_(self.lin4.weight)
        self.lin4.bias.data.fill_(0.01)
        
    def forward(self, input):
        out = F.relu(self.lin1(input))
        out = F.relu(self.lin2(out))
#        out = F.relu(self.lin3(out))
        out = self.lin4(out)
        #out = self.Sigmoid(self.lin3(out))
        return out
    
    
    
class LinearClassifier(nn.Module):
    
    def __init__(self, in_dim, output_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.linear3 = nn.Linear(in_dim, output_size)

#        init.xavier_uniform(self.linear1.weight)
#        self.linear1.bias.data.fill_(0.01)
#        init.xavier_uniform(self.linear2.weight)
#        self.linear2.bias.data.fill_(0.01)
        
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        out = self.linear3(out)
        out = F.softmax(out, dim=1)
        return out
    
    
