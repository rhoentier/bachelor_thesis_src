import torchvision
from torch import nn

from ai_framework.models.models import InceptionNet3, InceptionNet3Gray
from ai_framework.traffic_sign_main import TrafficSignMain
from ai_framework.ts_ai import TSAI


###
#
# Taken from:
# Johannes Alecke. Analyse und Optimierung von Angriffen auf tiefe neuronale Netze, Hochschule Bonn-Rhein-Sieg, 2020
#
###


# https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
def set_mobile_net_v2_module():
    module = torchvision.models.mobilenet_v2(pretrained=True)
    num_features = module.classifier[1].in_features
    module.classifier[1] = nn.Linear(num_features, 43)
    return module


# https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
def set_resnext_module():
    module = torchvision.models.resnext50_32x4d(pretrained=True)
    num_features = module.fc.in_features
    module.fc = nn.Linear(num_features, 43)
    return module


# https://pytorch.org/hub/pytorch_vision_inception_v3/
def set_inception_v3_module():
    module = torchvision.models.inception_v3(pretrained=True)
    num_features = module.fc.in_features
    module.fc = nn.Linear(num_features, 43)
    return module


def load_pretrained_resnext():
    ai = TSAI("PreTrained_ResNext", net=set_resnext_module())
    main = TrafficSignMain(model=ai, epochs=15, image_size=224)
    main.loading_ai()
    return main


def load_pretrained_mobile_net_v2():
    ai = TSAI("PreTrained_MobileNetV2", net=set_mobile_net_v2_module())
    main = TrafficSignMain(model=ai, epochs=15, image_size=224)
    main.loading_ai()
    return main


def load_pretrained_inception_v3():
    ai = TSAI("PreTrained_InceptionV3", net=set_inception_v3_module())
    main = TrafficSignMain(model=ai, epochs=15, image_size=224)
    main.loading_ai()
    return main


def load_self_trained_inception_net3():
    ai = TSAI("SelfTrained_Model", net=InceptionNet3())
    main = TrafficSignMain(model=ai, epochs=15, image_size=32)
    main.loading_ai()
    return main


def load_self_trained_inception_net3_gray():
    ai = TSAI("SelfTrained_Model_Gray", net=InceptionNet3Gray())
    main = TrafficSignMain(model=ai, epochs=15, image_size=32)
    main.loading_ai()
    return main
