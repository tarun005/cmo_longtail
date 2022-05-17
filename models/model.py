import torch.nn as nn
import torch
import models.resnet as resnet

class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class Encoder(nn.Module):
    def __init__(self, arch, use_norm=False, bn_dim=256, total_classes=None):
        super(Encoder, self).__init__()
        self.model_fc = resnet.__dict__[arch](num_classes=total_classes, use_norm=use_norm)
        feature_len = self.model_fc.fc.in_features()
        self.model_fc.fc = nn.Identity()
        self.bottleneck_0 = nn.Linear(feature_len, bn_dim)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.BatchNorm1d(bn_dim), nn.ReLU())
        self.total_classes = total_classes
        if total_classes:
            self.classifier_layer = predictor(bn_dim, total_classes)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        if not self.total_classes:
            return (out_bottleneck, None)
        logits = self.classifier_layer(out_bottleneck)
        return (out_bottleneck, logits)

    # def get_parameters(self): 
    #     parameter_list = [{"params": self.model_fc.parameters(), "lr_mult": 0.1}, \
    #                       {"params": self.bottleneck_layer.parameters(), "lr_mult": 1}]

    #     return parameter_list


class discriminatorDANN(nn.Module):
    def __init__(self, feature_len):
        super(discriminatorDANN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(feature_len, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)

        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3)

    def forward(self, x, y):
        f2 = self.fc1(x)
        f = self.fc2_3(f2)
        return f


class discriminatorCDAN(nn.Module):
    def __init__(self, feature_len, total_classes):
        super(discriminatorCDAN, self).__init__()

        self.ad_layer1 = nn.Linear(feature_len * total_classes, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3)

    def forward(self, x, y):
        op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        ad_in = op_out.view(-1, y.size(1) * x.size(1))
        f2 = self.fc1(ad_in)
        f = self.fc2_3(f2)
        return f