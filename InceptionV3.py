import torch
import torch.nn as nn
import torchvision

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()

    def forward(self):
        return out

if __name__=='__main__':
    model = InceptionV3()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)