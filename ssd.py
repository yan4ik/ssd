import torch
import torch.nn as nn
import torch.nn.functional as F


vgg_base = {
    '300': [64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'C', 
            512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'C', 
            512, 512, 512, 'M',
            512, 512, 512],
}

extras = {
    '300': [256, 'S', 
            512, 128, 'S', 
            256, 128, 256, 128, 256],
    '512': [256, 'S', 
            512, 128, 'S', 
            256, 128, 'S', 
            256, 128, 'S', 
            256],
}

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def vgg(cfg, in_channels, batch_norm=False):

    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, 
               conv6,
               nn.ReLU(inplace=True), 
               conv7, 
               nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg, in_channels, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    
    layers = []
    flag = False

    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    
    return layers


#def add_extras(cfg, i, batch_norm=False, size=300):
#    # Extra layers added to VGG for feature scaling


class SSD(nn.Module):
    """
    Single Shot Multibox Architecture

    The network is composed of a base VGG network followed by the
    added multibox conv layers. Each multibox layer branches into:
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """


    def __init__(self, base, extras, head, num_classes, size):
        
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)

        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()


    def forward(self, x, test=False):
        """
        Applies network layers and ops on input image(s) x.
        """

        sources = []
        loc = []
        conf = []

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = F.normalize(x, p=2, dim=1)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )

        return output


def multibox(vgg, extra_layers, cfg, num_classes):

    loc_layers = []
    conf_layers = []

    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, 
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, 
                                  cfg[k] * num_classes,  kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_net(size=300, num_classes=2):
    
    if size != 300 and size != 512:
        
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return


    return SSD(*multibox(vgg(vgg_base[str(size)], 3),
                         add_extras(extras[str(size)], 1024, size=size),
                         mbox[str(size)],
                         num_classes),
               num_classes=num_classes,
               size=size)
