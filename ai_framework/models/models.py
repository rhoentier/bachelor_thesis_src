import torch
from torch import nn


###
#
# Taken from:
# Johannes Alecke. Analyse und Optimierung von Angriffen auf tiefe neuronale Netze, Hochschule Bonn-Rhein-Sieg, 2020
#
###


class InceptionNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self.define_features()
        self.classifiers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 43),
        )

    def forward(self, input_val):
        x = self.features(input_val)
        x = x.view(-1, 256 * 1 * 1)
        x = self.classifiers(x)
        return x

    @staticmethod
    def define_features():
        in_channels = 3
        inception = InceptionA(in_channels)

        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        batch_conv1 = BatchConv(256, 256, kernel_size=2, name="batch_conv1")
        batch_conv2 = BatchConv(256, 256, kernel_size=2, name="batch_conv2")
        batch_conv3 = BatchConv(256, 256, kernel_size=2, name="batch_conv3")

        layers = [inception, pool1, batch_conv1, pool2, batch_conv2, pool3, batch_conv3, pool4]
        return nn.Sequential(*layers)


class InceptionNet3Gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = self.define_features()
        self.classifiers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 43),
        )

    def forward(self, input_tensor):
        input_tensor = 0.2989 * input_tensor[:, 0, :, :] + 0.5870 * input_tensor[:, 1, :, :] + 0.1140 * \
                       input_tensor[:, 2, :, :]
        input_tensor = input_tensor.unsqueeze(1)
        x = self.features(input_tensor)
        x = x.view(-1, 256 * 1 * 1)
        x = self.classifiers(x)
        return x

    @staticmethod
    def define_features():
        in_channels = 1
        inception = InceptionA(in_channels)

        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        batch_conv1 = BatchConv(256, 256, kernel_size=2, name="batch_conv1")
        batch_conv2 = BatchConv(256, 256, kernel_size=2, name="batch_conv2")
        batch_conv3 = BatchConv(256, 256, kernel_size=2, name="batch_conv3")

        layers = [inception, pool1, batch_conv1, pool2, batch_conv2, pool3, batch_conv3, pool4]
        return nn.Sequential(*layers)


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # parallel_dummy layers are just used for the investigator to detect new parallel paths
        self.parallel_dummyA = NewParallelChainDummy()
        self.conv1x1 = BatchConv(in_channels, 64, kernel_size=1)

        self.parallel_dummyB = NewParallelChainDummy()
        self.conv5x5_1 = BatchConv(in_channels, 48, kernel_size=1)
        self.conv5x5_2 = BatchConv(48, 64, kernel_size=5, padding=2)

        self.parallel_dummyC = NewParallelChainDummy()
        self.conv3x3dbl_1 = BatchConv(in_channels, 64, kernel_size=1)
        self.conv3x3dbl_2 = BatchConv(64, 96, kernel_size=3, padding=1)
        self.conv3x3dbl_3 = BatchConv(96, 96, kernel_size=3, padding=1)

        self.parallel_dummyD = NewParallelChainDummy()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool1x1 = BatchConv(in_channels, 32, kernel_size=1)

        self.parallel_dummyE = NewParallelChainDummy()
        self.cat = Cat(concat_dim=1, dims=[64, 64, 96, 32])

    def forward(self, input_tensor):
        _ = self.parallel_dummyA(input_tensor)
        conv1x1 = self.conv1x1(input_tensor)

        _ = self.parallel_dummyB(input_tensor)
        conv5x5 = self.conv5x5_1(input_tensor)
        conv5x5 = self.conv5x5_2(conv5x5)

        _ = self.parallel_dummyC(input_tensor)
        conv3x3dbl = self.conv3x3dbl_1(input_tensor)
        conv3x3dbl = self.conv3x3dbl_2(conv3x3dbl)
        conv3x3dbl = self.conv3x3dbl_3(conv3x3dbl)

        _ = self.parallel_dummyD(input_tensor)
        branch_pool = self.pool(input_tensor)
        branch_pool = self.pool1x1(branch_pool)

        _ = self.parallel_dummyE(input_tensor)
        output = [conv1x1, conv5x5, conv3x3dbl, branch_pool]
        output = self.cat(output)
        # output = torch.cat(output, 1)
        return output


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_features_1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.extract_features_2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 400, 5, 1, 0)
        )

        self.dropout = nn.Dropout(p=0)
        self.classifier = nn.Linear(1576, 43)

    def forward(self, input_tensor):
        x = self.extract_features_1(input_tensor)
        feature_1 = x
        feature_2 = self.extract_features_2(x)

        feature_1 = feature_1.view(-1, 1176)
        feature_2 = feature_2.view(-1, 400)

        feature_s = torch.cat([feature_1, feature_2], 1)
        feature_s = self.dropout(feature_s)
        x = self.classifier(feature_s)
        return x


class Cat(nn.Module):
    def __init__(self, concat_dim=1, dims=None):
        super(Cat, self).__init__()
        if dims is None:
            dims = []
        self.concat_dim = concat_dim
        self.dims = dims

    def forward(self, input_list):
        output = torch.cat(input_list, self.concat_dim)
        return output


class NewParallelChainDummy(nn.Module):
    def __init__(self):
        super(NewParallelChainDummy, self).__init__()
        pass

    @staticmethod
    def forward(x):
        return x


class BatchConv(nn.Module):
    def __init__(self, in_channels, out_channels, name="Example", **kwargs):
        super(BatchConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.name = name

    def forward(self, input_val):
        output = self.conv(input_val)
        output = self.bn(output)
        output = self.relu(output)
        return output
