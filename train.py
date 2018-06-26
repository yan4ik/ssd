from pathlib import Path

import torch
from torch.autograd import Variable
import torch.utils.data as data
import torch.optim as optim

from ssd import build_net
from dataset import PedestrianDataset, preproc, detection_collate
from box import PriorBox
from multibox_loss import multibox_loss
from config import WIDER_300, WIDER_512


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


img_dim = 512
rgb_means = (104, 117, 123)
num_classes = 3

pedestrian_path = Path("/mnt/pedestrian/data")
pedestrian_dataset = PedestrianDataset(pedestrian_path,
                                       preproc=preproc(img_dim, 
                                                       rgb_means))

pedestrian_dataloader = data.DataLoader(pedestrian_dataset, 
                                        batch_size=15,
                                        shuffle=True,
                                        collate_fn=detection_collate)

cfg = WIDER_512
priorbox = PriorBox(cfg)
priors = priorbox.forward()
priors = priors.cuda()

net = build_net(size=512, num_classes=num_classes)
net.base.load_state_dict(torch.load("vgg16_reducedfc.pth"))
net.cuda()

optimizer = optim.Adam(net.parameters(), lr=3e-4,  weight_decay=1e-4)

loc_losses = AverageMeter()
clf_losses = AverageMeter()

net.train()
for epoch in range(15):

    for batch_id, (item, target) in enumerate(pedestrian_dataloader):
        optimizer.zero_grad()

        if sum(t.numel() for t in target) == 0:
            print("Small batch, skipping ...")
            continue
        
        item, target = item.cuda(), [t.cuda() for t in target]
        item.requires_grad = True

        out = net(item)

        loc_loss, clf_loss = multibox_loss(out, priors, target)
        (loc_loss + clf_loss).backward()
        
        optimizer.step()

        loc_losses.update(loc_loss.item())
        clf_losses.update(clf_loss.item())

        if batch_id % 25 == 0:
            print(loc_losses.avg, clf_losses.avg)

    print("Finished epoch. Last batch:", len(target))

torch.save(net.cpu().state_dict(), "model.th")
