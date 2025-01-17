# Author: João Fernando Mari
# joaofmari.github.io
# https://github.com/joaofmari

from torchvision import models
import torch.nn as nn 

### import timm

def create_model(arch, ft, num_classes, bce):

    input_size = 224

    # **** torchvision ********************************************************
    if arch == 'alexnet': # OK 
        print(f'\nModel: AlexNet')
        if ft:
            model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        else:
            model = models.alexnet(weights=None)

        num_ftrs = model.classifier[6].in_features # alexnet: 4096

        if num_classes == 2 and bce:
            # Classificação binária:
            # Apenas 1 neurônio na camada de saída e função de perda (loss) BCELoss.
            model.classifier[6] = nn.Linear(num_ftrs, 1)
        else:
            # Classicação não binária: 
            # Número de neurônios na camada de sáida igual ao número de classes. 
            # Função de perda CrossEntropyLoss.
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1] # OK

    elif arch == 'vgg11':
        print('\nModel: VGG11_BN')
        if ft:
            model = models.vgg11_bn(weights='VGG11_BN_Weights.IMAGENET1K_V1')
        else:
            model = models.vgg11_bn(weights=None)

        num_ftrs = model.classifier[6].in_features # vgg11_bn: 4096

        if num_classes == 2 and bce:
            model.classifier[6] = nn.Linear(num_ftrs, 1)
        else:
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    elif arch == 'vgg16':
        print(f'\nModel: VGG16_BN')
        if ft:
            model = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        else:
            model = models.vgg16_bn(weights=None)

        num_ftrs = model.classifier[6].in_features # vgg16_bn: 4096

        if num_classes == 2 and bce:
            model.classifier[6] = nn.Linear(num_ftrs, 1)
        else:
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    # ====> ResNet
    # (resnet18, resnet34, resnet50, resnet100, resnet152)
    elif arch == 'resnet18': # OK 
        print('\nModel: ResNet18')
        if ft:
            model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet18(weights=None)

        num_ftrs = model.fc.in_features # resnet18: 512

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]] # OK

    elif arch == 'resnet50': # OK 
        print('\nModel: ResNet (resnet50)')
        if ft:
            model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet50(weights=None)

        num_ftrs = model.fc.in_features # resnet18: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]] # OK

    elif arch == 'resnet101': # OK 
        print('\nModel: ResNet (resnet101)')
        if ft:
            model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet101(weights=None)

        num_ftrs = model.fc.in_features # resnet18: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]] # OK

    elif arch == 'resnet152': # OK 
        print('\nModel: ResNet (resnet152)')
        if ft:
            model = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet152(weights=None)

        num_ftrs = model.fc.in_features # resnet18: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]] # OK

    elif arch == 'squeezenet':
        print('\nModel: SqueezeNet1_0')
        if ft:
            model = models.squeezenet1_0(weights='SqueezeNet1_0_Weights.IMAGENET1K_V1')
        else:
            model = models.squeezenet1_0(weights=None)

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        else:
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        # Grad-cam
        target_layers = model.features[-1] # Defined by joaofmari!

    elif arch == 'densenet':
        print('\nModel: DenseNet121')
        if ft:
            model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        else:
            model = models.densenet121(weights=None)

        num_ftrs = model.classifier.in_features # Densenet121: 1024

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(num_ftrs, 1)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    elif arch == 'inception':
        print('\nModel: Inception_V3')
        if ft:
            model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        else:
            model = models.inception_v3(weights=None)

        num_ftrs1 = model.AuxLogits.fc.in_features # Inception_v3: 768
        num_ftrs2 = model.fc.in_features # Inception_v3: 2048

        if num_classes == 2 and bce:
            model.AuxLogits.fc = nn.Linear(num_ftrs1, 1)
            model.fc = nn.Linear(num_ftrs2, 1)
        else:
            model.AuxLogits.fc = nn.Linear(num_ftrs1, num_classes)
            model.fc = nn.Linear(num_ftrs2, num_classes)

        # OBS.: A entrada da rede Inception possui tamanho 299x299
        input_size = 299
        
        # Grad-cam (TODO)
        target_layers = None

    # **** torchvision 2 ******************************************************
    # ====> VisionTransformer
    # (vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14)
    elif arch == 'vit_b_16': # OK
        print('\nModel: VisionTransformer (vit_b_16)')

        if ft:
            model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        else:
            model = models.vit_b_16(weights=None)

        num_ftrs = model.heads.head.in_features # vit_b_16: 768

        if num_classes == 2 and bce:
            model.heads.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam
        target_layers = model.encoder.layers.encoder_layer_11.mlp[-1] # Não testei!!!
        # Grad-cam #2
        target_layers = model.encoder.ln # Não testei!!!

    elif arch == 'vit_b_32': # OK
        print('\nModel: VisionTransformer (vit_b_32)')

        if ft:
            model = models.vit_b_32(weights='ViT_B_32_Weights.IMAGENET1K_V1')
        else:
            model = models.vit_b_32(weights=None)

        num_ftrs = model.heads.head.in_features # vit_b_32: 768

        if num_classes == 2 and bce:
            model.heads.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam
        target_layers = model.encoder.layers.encoder_layer_11.mlp[-1] # Não testei!!!
        # Grad-cam #2
        target_layers = model.encoder.ln # Não testei!!!

    elif arch == 'vit_l_16': # OK
        print('\nModel: VisionTransformer (vit_l_16)')

        if ft:
            model = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_V1')
        else:
            model = models.vit_l_16(weights=None)

        num_ftrs = model.heads.head.in_features # vit_l_16: 1024

        if num_classes == 2 and bce:
            model.heads.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam
        target_layers = model.encoder.layers.encoder_layer_23.mlp[-1] # Não testei!!!
        # Grad-cam #2
        target_layers = model.encoder.ln # Não testei!!!

    elif arch == 'vit_l_32': # OK
        print('\nModel: VisionTransformer (vit_l_32)')

        if ft:
            model = models.vit_l_32(weights='ViT_L_32_Weights.IMAGENET1K_V1')
        else:
            model = models.vit_l_32(weights=None)

        num_ftrs = model.heads.head.in_features # vit_l_32: 1024

        if num_classes == 2 and bce:
            model.heads.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam #1
        target_layers = model.encoder.layers.encoder_layer_23.mlp[-1] # Não testei!!!
        # Grad-cam #2
        target_layers = model.encoder.ln # Não testei!!!

    # ====> EfficientNet
    # (efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7)
    elif arch == 'efficientnet_b4': # OK
        print('\nModel: EfficientNet (efficientnet_b4)')

        if ft:
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_b4(weights=None)

        num_ftrs = model.classifier[1].in_features # efficientnet_b4: 1792

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[8][2] # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    # ====> EfficientNetV2
    # (efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l)
    elif arch == 'efficientnet_v2_s': #OK
        print('\nModel: EfficientNet (efficientnet_v2_s)')

        if ft:
            model = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_v2_s(weights=None)

        num_ftrs = model.classifier[1].in_features # efficientnet_b4: 1280

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[7][2] # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'efficientnet_v2_m': # OK
        print('\nModel: EfficientNet (efficientnet_v2_m)')

        if ft:
            model = models.efficientnet_v2_m(weights='EfficientNet_V2_M_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_v2_m(weights=None)

        num_ftrs = model.classifier[1].in_features # efficientnet_v2_m: 1280

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[8][2] # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'efficientnet_v2_l': # OK
        print('\nModel: EfficientNet (efficientnet_v2_l)')

        if ft:
            model = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_v2_l(weights=None)

        num_ftrs = model.classifier[1].in_features # efficientnet_v2_l: 1280

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[8][2] # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    # ====> MobileNet V3
    # (mobilenet_v3_small, mobilenet_v3_large)
    elif arch == 'mobilenet_v3_large':
        print('\nModel: MobileNet V3 (mobilenet_v3_large)')

        if ft:
            model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1')
        else:
            model = models.mobilenet_v3_large(weights=None)

        num_ftrs = model.classifier[3].in_features # mobilenet_v3_large: 1280

        if num_classes == 2 and bce:
            model.classifier[3] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[3] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[16][2]
        # Grad-cam. #2
        target_layers = model.avgpool

    # ====> ConvNext
    # (convnext_tiny, convnext_small, convnext_base, convnext_large)
    elif arch == 'convnext_tiny': # OK
        print('\nModel: ConvNeXt (convnext_tiny)')
        
        if ft:
            model = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        else:
            model = models.convnext_tiny(weights=None)

        num_ftrs = model.classifier[2].in_features # convnext_tiny: 768

        if num_classes == 2 and bce:
            model.classifier[2] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[2] = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam. #1
        target_layers = model.features[7][2].stochastic_depth # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'convnext_small': # OK
        print('\nModel: ConvNeXt (convnext_small)')
        
        if ft:
            model = models.convnext_small(weights='ConvNeXt_Small_Weights.IMAGENET1K_V1')
        else:
            model = models.convnext_small(weights=None)

        num_ftrs = model.classifier[2].in_features # convnext_small: 768

        if num_classes == 2 and bce:
            model.classifier[2] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[2] = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam. #1
        target_layers = model.features[7][2].stochastic_depth # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'convnext_base': # OK
        print('\nModel: ConvNeXt (convnext_base)')
        
        if ft:
            model = models.convnext_base(weights='ConvNeXt_Base_Weights.IMAGENET1K_V1')
        else:
            model = models.convnext_base(weights=None)

        num_ftrs = model.classifier[2].in_features # convnext_base: 768

        if num_classes == 2 and bce:
            model.classifier[2] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[2] = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam. #1
        target_layers = model.features[7][2].stochastic_depth # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'convnext_large': # OK
        print('\nModel: ConvNeXt (convnext_large)')
        
        if ft:
            model = models.convnext_large(weights='ConvNeXt_Large_Weights.IMAGENET1K_V1')
        else:
            model = models.convnext_large(weights=None)

        num_ftrs = model.classifier[2].in_features # convnext_large: 1536

        if num_classes == 2 and bce:
            model.classifier[2] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[2] = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam. #1
        target_layers = model.features[7][2].stochastic_depth # Não testei!!!
        # Grad-cam. #2
        target_layers = model.avgpool # Não testei!!!

    # ====> SwinTransformer
    # (swin_t, swin_s, swin_b)
    elif arch == 'swin_t': # OK
        print('\nModel: SwinTransformer (swin_t)')
        
        if ft:
            model = models.swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_t(weights=None)

        num_ftrs = model.head.in_features # swin_t: 768

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'swin_s': # OK
        print('\nModel: SwinTransformer (swin_s)')
        
        if ft:
            model = models.swin_s(weights='Swin_S_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_s(weights=None)

        num_ftrs = model.head.in_features # swin_s: 768

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'swin_b': # OK
        print('\nModel: SwinTransformer (swin_b)')
        
        if ft:
            model = models.swin_b(weights='Swin_B_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_b(weights=None)

        num_ftrs = model.head.in_features # swin_b: 1024

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    # ====> SwinTransformer V2
    # (swin_v2_t, swin_v2_s, swin_v2_b)
    elif arch == 'swin_v2_t': # OK
        print('\nModel: SwinTransformer V2 (swin_v2_t)')

        if ft:
            model = models.swin_v2_t(weights='Swin_V2_T_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_v2_t(weights=None)

        num_ftrs = model.head.in_features # swin_v2_t: 768

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'swin_v2_s': # OK
        print('\nModel: SwinTransformer V2 (swin_v2_s)')

        if ft:
            model = models.swin_v2_s(weights='Swin_V2_S_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_v2_s(weights=None)

        num_ftrs = model.head.in_features # swin_v2_s: 768

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'swin_v2_b': # OK
        print('\nModel: SwinTransformer V2 (swin_v2_b)')

        if ft:
            model = models.swin_v2_b(weights='Swin_V2_B_Weights.IMAGENET1K_V1')
        else:
            model = models.swin_v2_b(weights=None)

        num_ftrs = model.head.in_features # swin_v2_b: 1024

        if num_classes == 2 and bce:
            model.head = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.head = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.norm # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    # ====> ResNeXt
    # (resnext50_32x4d, resnext101_32x8d, resnext101_64x4d)
    elif arch == 'resnext50_32x4d': # OK
        print('\nModel: ResNeXt (resnext50_32x4d)')

        if ft:
            model = models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V1')
        else:
            model = models.resnext50_32x4d(weights=None)

        num_ftrs = model.fc.in_features # resnext50_32x4d: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.layer4[2].relu # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'resnext101_32x8d': # OK
        print('\nModel: ResNeXt (resnext101_32x8d)')

        if ft:
            model = models.resnext101_32x8d(weights='ResNeXt101_32X8D_Weights.IMAGENET1K_V1')
        else:
            model = models.resnext101_32x8d(weights=None)

        num_ftrs = model.fc.in_features # resnext101_32x8d: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.layer4[2].relu # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!

    elif arch == 'resnext101_64x4d': # OK
        print('\nModel: ResNeXt (resnext101_64x4d)')

        if ft:
            model = models.resnext101_64x4d(weights='ResNeXt101_64X4D_Weights.IMAGENET1K_V1')
        else:
            model = models.resnext101_64x4d(weights=None)

        num_ftrs = model.fc.in_features # resnext101_64x4d: 2048

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        input_size = 224

        # Grad-cam #1
        target_layers = model.layer4[2].relu # Não testei!!!
        # Grad-cam #2
        target_layers = model.avgpool # Não testei!!!


    return model, input_size, target_layers