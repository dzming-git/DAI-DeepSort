# deep_sort/deep_sort/deep/feature_extractor.py
# 添加指定gpu的功能

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List, Tuple
from torch import Tensor

from deep_sort.deep_sort.deep.model import Net

class FeatureExtractor(object):
    def __init__(self, model_path: str, device_str: str) -> None:
        self.device_str: str = device_str
        self.device: torch.device = torch.device(device_str)
        self.net: Net = Net(reid=True)
        state_dict: dict = torch.load(model_path, map_location=self.device)['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size: Tuple[int, int] = (64, 128)
        self.norm: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops: List[np.ndarray]) -> Tensor:
        def _resize(im: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
            return cv2.resize(im, size)

        im_batch: Tensor = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops: List[np.ndarray]) -> np.ndarray:
        im_batch: Tensor = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features: Tensor = self.net(im_batch)
        return features.cpu().numpy()
