import os
import pickle

import anytree
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO as pyCOCO
import json
from utils import load, save

from .utils import download_url, extract_file, make_classes_counts, make_flat_index, make_tree

class COCO(Dataset):
    data_name = "COCO"
    file = [
        (
            "http://images.cocodataset.org/zips/train2017.zip",
            None,
            
        ),
        (
            "http://images.cocodataset.org/zips/val2017.zip",
            None,
        ),
        (
            "http://images.cocodataset.org/zips/test2017.zip",
            None,
        ),
        (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            None,
        )
    ]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not os.path.exists(self.processed_folder):
            self.process()
        id, self.data, self.target = load(
            os.path.join(self.processed_folder, "{}.pt".format(self.split)),
            mode="pickle",
        )
        self.classes_counts = self.make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, "meta.pt"), mode="pickle")
        self.other = {"id": id}
        # self.target = [t + [-1] * (len(self.classes_counts) - len(t)) for t in self.target]
       
    def __getitem__(self, index):
        data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, "data": data, "target": target}
        if self.transform is not None:
            input = self.transform(input)
        return input
    
    def __len__(self):
        return len(self.data)
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")
    
    def process(self):
        if not os.path.exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, "train.pt"), mode="pickle")
        save(test_set, os.path.join(self.processed_folder, "test.pt"), mode="pickle")
        save(meta, os.path.join(self.processed_folder, "meta.pt"), mode="pickle")
        return
    
    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        for url, md5 in self.file:
            filename = url.rpartition("/")[2]
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return 

    def __repr__(self):
        fmt_str = "Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}".format(
            self.__class__.__name__,
            self.__len__(),
            self.root,
            self.split,
            self.transform.__repr__(),
        )
        return fmt_str
    
    def make_data(self):
        train_coco = pyCOCO(os.path.join(self.raw_folder, "annotations", "instances_train2017.json"))
        val_coco = pyCOCO(os.path.join(self.raw_folder, "annotations", "instances_val2017.json"))


        train_img_ids = train_coco.getImgIds()
        val_img_ids = val_coco.getImgIds()

        train_data = [train_coco.loadImgs(img_id)[0]['file_name'] for img_id in train_img_ids]
        val_data = [val_coco.loadImgs(img_id)[0]['file_name'] for img_id in val_img_ids]

        train_target = [train_coco.getAnnIds(imgIds=[img_id]) for img_id in train_img_ids]
        train_target = [train_coco.loadAnns(ann_ids) for ann_ids in train_target]
        val_target = [val_coco.getAnnIds(imgIds=[img_id]) for img_id in val_img_ids]
        val_target = [val_coco.loadAnns(ann_ids) for ann_ids in val_target]

        train_id = np.arange(len(train_data)).astype(np.int64)
        val_id = np.arange(len(val_data)).astype(np.int64)

        with open(os.path.join(self.raw_folder, "annotations", "instances_train2017.json"), "r") as f:
            data = json.load(f)
            classes_to_labels = {category['name']: category['id'] for category in data['categories']}
        
        target_size = len(classes_to_labels)

        # train_target = [target + [-1]* (target_size - len(target)) for target in train_target]
        # val_target = [target + [-1]* (target_size - len(target)) for target in val_target]

        return (
            (train_id, train_data, train_target),
            (val_id, val_data, val_target),
            (classes_to_labels, target_size)
        )

    def make_classes_counts(self, targets):
        counts = {}
        for anns in targets:
            for ann in anns:
                category_id = ann['category_id']
                if category_id in counts:
                    counts[category_id] += 1
                else:
                    counts[category_id] = 1
        return counts
        
        
        