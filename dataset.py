import torch
import torch.utils.data as data
import numpy as np
import cv2

from utils import open_image


class preproc:


    def __init__(self, resize, rgb_means):

        self.resize = resize
        self.means = rgb_means


    def final_preproc_image(self, image):

        image = cv2.resize(image,
                           (self.resize, self.resize))

        image -= self.means

        return image.transpose(2, 0, 1)
                           


    def __call__(self, image, target):

        if len(target) == 0:
            image = self.final_preproc_image(image)
            
            return image, target

        target = target.copy()

        height, width, _ = image.shape

        # normalizing bounding boxes to be in [0, 1]        
        target[:, 0:4:2] /= width
        target[:, 1:4:2] /= height

        # add data augmentation
        # which we skip for now

        image = self.final_preproc_image(image)

        return image, target


class PedestrianDataset(data.Dataset):

    
    @staticmethod
    def parse_annotations_line(line):
        """
        Return image_name, list of annotations for that image (one image can have multiple annotations).

        One annotation is a numpy array (x_upper_left, y_upper_left, x_lower_right, y_lower_right, label)
        """

        img_name, *annotations_raw = line.split()
        annotations_raw = list(map(int, annotations_raw))

        annotations = []
        for i in range(0, len(annotations_raw), 5):
        
            label, bbox = annotations_raw[i], annotations_raw[i+1:i+5]
            # In the file bounding boxes are given as (x_upper_left, y_upper_left, width, height).
            # We convert them into (x_upper_left, y_upper_left, x_down_right, y_down_right)
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            annotations.append(bbox + [label])

        return img_name, np.array(annotations, dtype=np.float32)


    def __init__(self, root, preproc=None):
        
        self.train_dir = root / "train"
        self.annotation_file = root / "train_annotations.txt"
        
        with open(root / "train.txt") as rf:
            self.files = [line.rstrip() for line in rf]

        filename2id = {f: i for i, f in enumerate(self.files)}

        self.labels = [[] for _ in self.files]
        with open(self.annotation_file) as rf:
            for img_name, annotations in map(self.parse_annotations_line, rf):
                self.labels[filename2id[img_name]] = annotations

        self.preproc = preproc


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        _file = self.files[index]
        file_path = self.train_dir / _file
        img = open_image(file_path)

        target = self.labels[index]

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), torch.from_numpy(target)


def detection_collate(batch):

    imgs, labels = zip(*batch)
    
    return torch.stack(imgs, 0), labels
