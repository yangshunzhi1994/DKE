from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, \
    resnet20_aux, resnet56_aux, resnet8x4_aux, resnet32x4_aux
from .resnetv2 import ResNet50, ResNet50_aux
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_40_2_aux, wrn_16_2_aux, wrn_40_1_aux
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn, vgg13_bn_aux, vgg8_bn_aux
from .mobilenetv2 import mobile_half, mobilenetV2_aux
from .ShuffleNetv1 import ShuffleV1, ShuffleV1_aux
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_aux
from .teacherNet import Teacher
from .studentNet import CNN_RIS

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet20_aux': resnet20_aux,
    'resnet56_aux': resnet56_aux,
    'resnet8x4_aux': resnet8x4_aux,
    'resnet32x4_aux': resnet32x4_aux,
    'ResNet50': ResNet50,
    'ResNet50_aux': ResNet50_aux,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_40_2_aux': wrn_40_2_aux,
    'wrn_16_2_aux': wrn_16_2_aux,
    'wrn_40_1_aux': wrn_40_1_aux,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg13_bn_aux': vgg13_bn_aux,
    'vgg8_bn_aux': vgg8_bn_aux,
    'MobileNetV2': mobile_half,
    'MobileNetV2_aux': mobilenetV2_aux,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV1_aux': ShuffleV1_aux,
    'ShuffleV2_aux': ShuffleV2_aux,
    'Teacher': Teacher,
    'CNN_RIS': CNN_RIS,
}
