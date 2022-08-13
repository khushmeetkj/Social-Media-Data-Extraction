import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
class MultiHeadResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layers according to the number of categories
        self.l0 = nn.Linear(2048, 5) # for gender
        self.l1 = nn.Linear(2048, 7) # for masterCategory
        self.l2 = nn.Linear(2048, 45) # for subCategory
        self.l3 = nn.Linear(2048, 143) # for articleType
        self.l4 = nn.Linear(2048, 47) # for baseColour
        self.l5 = nn.Linear(2048, 4) # for seaon
        self.l6 = nn.Linear(2048, 9) # for usage
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        l4 = self.l4(x)
        l5 = self.l5(x)
        l6 = self.l6(x)
        return l0, l1, l2, l3, l4, l5, l6
        