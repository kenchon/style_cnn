import torch
import torch.nn as nn

N_tags = 66

class Stylenet(nn.Module):
    def __init__(self):
        super(Stylenet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1))
        self.conv2 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1))
        self.conv2_drop = nn.Dropout(0.25)
        self.pool1 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn1 = nn.BatchNorm2d(64,0.001,0.9,True)
        self.conv3 = nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1))
        self.conv4 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1))
        self.conv4_drop = nn.Dropout(0.25),
        self.pool2 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn2 = nn.BatchNorm2d(128,0.001,0.9,True)
        self.conv5 = nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1))
        self.conv6 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.conv6_drop = nn.Dropout(0.25)
        self.pool3 =nn.MaxPool2d((4, 4),(4, 4))
        self.bn3 = nn.BatchNorm2d(256,0.001,0.9,True)
        self.conv7 = nn.Conv2d(256,128,(1, 1))
        self.linear1 = nn.Linear(3072,128)
        self.linear2 = nn.Linear(128, N_tags)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input):
        x = nn.ReLU(self.conv1(input))
        x = nn.ReLU(self.conv2(input))
        x = self.conv2_drop(x)
        x = self.bn1(self.pool1(x))
        x = nn.ReLU(self.conv3(x))
        x = self.conv4_drop(nn.ReLU(self.conv4(x)))
        x = self.bn2(self.pool2(x))
        x = nn.ReLU(self.conv5(x))
        x = nn.ReLU(self.conv6(x))
        x = self.conv6_drop(x)
        x = self.bn3(self.pool3(x))
        x = nn.ReLU(self.conv7(x))
        x_middle = self.linear1(x)
        x = self.linear2(x_middle)
        x = self.linear1(x)
        x = self.logsoftmax(x)
        return x, x_middle

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

modelA = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.MaxPool2d((4, 4),(4, 4)),
	nn.BatchNorm2d(64,0.001,0.9,True),
	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.MaxPool2d((4, 4),(4, 4)),
	nn.BatchNorm2d(128,0.001,0.9,True),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.MaxPool2d((4, 4),(4, 4)),
	nn.BatchNorm2d(256,0.001,0.9,True),
	nn.Conv2d(256,128,(1, 1)),
	nn.ReLU(),
	Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(3072,128)) # Linear,
    #nn.Linear(128, 40)
)

modelA.load_state_dict(torch.load("stylenet.pth"))
modelB = Stylenet()

from collections import OrderedDict

# Get Sequential state dict
state_dict = modelA.state_dict()

for keyA, keyB in zip(modelA.state_dict(), modelB.state_dict()):
    print('Changing {} to {}'.format(keyA, keyB))
    state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())

state_dict['linear2.weight'] = torch.ones([N_tags, 128])
state_dict['linear2.bias'] = torch.ones([N_tags])

# state_dict should keep the old values with new keys
modelB.load_state_dict(state_dict)

if __name__ == "__main__":
    print("model {}".format(modelB))
