from math import sqrt
import itertools as it

import torch

#if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class PriorBox:


    def __init__(self, cfg):

        super(PriorBox, self).__init__()
        
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.scales = cfg['scales']
        self.aspect_ratios = cfg['aspect_ratios']


    def forward(self):
        
        mean = []

        prev_size = 0

        for k, f in enumerate(self.feature_maps):
            for i, j in it.product(range(f), repeat=2):
                # centers of boxes normalized from 0 to 1
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                s_k = self.scales[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.scales[k + 1] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
            
            print("Feature map %s bouding box num: (%s, %s)" % (f, prev_size // 4, len(mean) // 4))
            prev_size = len(mean)

        # back to torch land
        output = torch.tensor(mean).view(-1, 4)
        output.clamp_(min=0, max=1)

        return output
