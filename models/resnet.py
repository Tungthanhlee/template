import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
import pretrainedmodels

class ResNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        """
        Keep the input features and modify the out features equivalent to num_class
        Delete the last linear layer and replace it to the customized layer
        """
        model = pretrainedmodels.__dict__[model_name](num_classes = 1000, pretrained="imagenet")
        in_features = model.last_linear.in_features
        del model.last_linear 
        feature_map = list(model.children())
        self.backbone = nn.Sequential(*list(feature_map))

        self.fc = nn.Linear(in_features, num_classes)
    
    def features(self, x):
        return self.backbone(x)

    def logits(self, x):
        return self.fc(x)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x= self.logits(x)
        return x
        

if __name__ == "__main__":
    model_name = "resnet34"
    num_classes = 10
    # model = pretrainedmodels.__dict__[model_name](num_classes = 1000, pretrained="imagenet")
    # in_features = model.last_linear.in_features
    # del model.last_linear
    # feature_map = list(model.children())
    # # print(feature_map)
    # backbone = nn.Sequential(*list(feature_map))
    # print(backbone)
    # fc = nn.Linear(in_features, num_classes)
    # print(fc)
    model = ResNet(model_name, num_classes)
    dum_input = torch.rand(3,224, 224, dtype=torch.float)
    dum_input = dum_input.unsqueeze(0)
    # print(dum_input.size())
    model.eval()
    output = model(dum_input)
    print(output)
    # out_features = model.features(dum_input)
    # print(out_features.size())
    # out_logit = model.logits(out_features)
    # print(out_logit)
    