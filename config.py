WIDER_300 = {
    # this feature maps occur in `sources` array - that means
    'feature_maps': [38, 19, 10, 5, 3, 1],
    # image size
    'min_dim': 300,
    'scales': [30, 60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'num_classes': 2 + 1
}

WIDER_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'scales': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'num_classes': 2 + 1
}
