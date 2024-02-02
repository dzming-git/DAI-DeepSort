from Deepsort.deep_sort import DeepSort
from Deepsort.deep_sort.deep.feature_extractor import Extractor
from Deepsort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from Deepsort.deep_sort.sort.tracker import Tracker
from Deepsort.deep_sort.deep.model import Net
import torchvision.transforms as transforms
import logging


import torch
import numpy as np
import queue

import copy
import threading
import traceback
from typing import Dict, List

import warnings
warnings.filterwarnings('always')

# step说明
# 0 完成检测图片
# 1 开始检测图片
# 2 完成添加图片
# 3 开始添加图片
# 4 完成添加uid
DETECT_IMAGE_COMPLETE = 0
DETECT_IMAGE_START = 1
ADD_IMAGE_COMPLETE = 2
ADD_IMAGE_START = 3
ADD_UID_COMPLETE = 4

class Parse:
    MAX_DIST = 0.2
    MIN_CONFIDENCE = 0.3
    NMS_MAX_OVERLAP = 0.5
    MAX_IOU_DISTANCE = 0.7
    MAX_AGE = 70
    N_INIT = 3
    NN_BUDGET = 100


def xyxy_to_xywh(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class MyExtractor(Extractor):
    def __init__(self, model_path, device):
        self.device = device
        self.net = Net(reid=True)
        state_dict = torch.load(model_path, map_location=torch.device(self.device))[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class DeepSortTracker(DeepSort):
    class ImgInfo:
        uid = 0
        image_shape = np.zeros((0, 0, 0))
        results: Dict[int, List[float]] = {}
        results_update: Dict[int, bool] = {}
        # results = []
        is_used = False
        step = ADD_UID_COMPLETE
        lock = threading.Lock()
    
    def __init__(self, device, weights):
        parse = Parse()
        self.min_confidence = parse.MIN_CONFIDENCE
        self.nms_max_overlap = parse.NMS_MAX_OVERLAP
        self.extractor = MyExtractor(weights, device)
        max_cosine_distance = parse.MAX_DIST
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, Parse.NN_BUDGET)
        self.tracker = Tracker(
            metric, max_iou_distance=parse.MAX_IOU_DISTANCE, max_age=parse.MAX_AGE, n_init=parse.N_INIT)
        
        # 图像、结果缓存
        self.max_tracking_length = 10
        self._max_cache = 100
        self._img_infos:Dict[int, DeepSortTracker.ImgInfo] = dict()
        self._img_uid_fifo:queue.Queue[int] = queue.Queue(maxsize=self._max_cache)

        # 最新检测完成的uid
        self.latest_detection_completed_uid = 0

    def check_uid_exist(self, uid) -> bool:
        return uid in self._img_infos

    def get_statue(self, uid) -> int:
        if not self.check_uid_exist(uid):
            # warnings.warn("该uid不存在", UserWarning)
            return -1
        return self._img_infos[uid].step
    
    def add_uid(self, uid) -> bool:
        if self.check_uid_exist(uid):
            warnings.warn("该uid已存在", UserWarning)
            return False
        # 清理溢出
        if (len(self._img_infos) >= self._max_cache):
            uid_rm = self._img_uid_fifo.get()
            if not self._img_infos[uid_rm].is_used:
                warnings.warn(f'弹出uid={uid_rm}-未被使用', UserWarning)
            with self._img_infos[uid_rm].lock:
                self._img_infos.pop(uid_rm)
        self._img_uid_fifo.put(uid)
        self._img_infos[uid] = DeepSortTracker.ImgInfo()
        self._img_infos[uid].step = ADD_UID_COMPLETE
        return True
    
    # 添加、检测
    def add_image_and_bboxes(self, uid, image, normalized_bboxes):
        if not self.check_uid_exist(uid):
            self.add_uid(uid)
        elif self._img_infos[uid].step != ADD_UID_COMPLETE:
            print(self._img_infos[uid].step)
            warnings.warn("重复添加", UserWarning)
            return False
        self._img_infos[uid].step = ADD_IMAGE_COMPLETE
        self._img_infos[uid].image_shape = image.shape
        result = None
        self._img_infos[uid].step = DETECT_IMAGE_START
        results = []
        if len(normalized_bboxes) != 0:
            xywh = []
            for normalized_bbox in normalized_bboxes:
                bbox = [
                    normalized_bbox[0] * image.shape[1],  # 左上角 x 坐标
                    normalized_bbox[1] * image.shape[0],  # 左上角 y 坐标
                    normalized_bbox[2] * image.shape[1],  # 右下角 x 坐标
                    normalized_bbox[3] * image.shape[0]   # 右下角 y 坐标
                ]
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(bbox)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh.append(xywh_obj)
            xywh = torch.Tensor(xywh)
            confss = torch.from_numpy(np.ones(shape=(len(normalized_bboxes))))
            results = super().update(xywh, confss, image)
        else:
            super().increment_ages()
        
        # 将全部id置为未更新
        for id in self._img_infos[uid].results_update:
            self._img_infos[uid].results_update[id] = False
        # 更新id的result
        for result in results:
            id = result[4]
            # 新的id
            if id not in self._img_infos[uid].results_update:
                self._img_infos[uid].results[id] = []
            # 判断结果数量
            if len(self._img_infos[uid].results[id]) > self.max_tracking_length:
                self._img_infos[uid].results[id] = self._img_infos[uid].results[id][1:]
            # 添加结果
            self._img_infos[uid].results[id].append([
                float(result[0]) / image.shape[1],
                float(result[1]) / image.shape[0],
                float(result[2]) / image.shape[1],
                float(result[3]) / image.shape[0],
            ])
            # 更新id状态
            self._img_infos[uid].results_update[id] = True
        # 删除未更新的id
        deleteId = []
        for id in self._img_infos[uid].results_update:
            update = self._img_infos[uid].results_update[id]
            if not update:
                deleteId.append(id)
        for id in deleteId:
            del self._img_infos[uid].results[id]
            del self._img_infos[uid].results_update[id]
        self._img_infos[uid].step = DETECT_IMAGE_COMPLETE
        self.latest_detection_completed_uid = uid
        return True

    def get_result_by_uid(self, uid) -> Dict[int, List[float]]:
        if not self.check_uid_exist(uid):
            warnings.warn("该uid不存在", UserWarning)
            return None
        if self._img_infos[uid].step != DETECT_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        with self._img_infos[uid].lock:
            self._img_infos[uid].is_used = True
            if len(self._img_infos[uid].results):
                print(self._img_infos[uid].results)
        return self._img_infos[uid].results
