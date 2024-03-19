from deep_sort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.sort.preprocessing import non_max_suppression
from deep_sort.deep_sort.sort.detection import Detection
from deep_sort.deep_sort.sort.tracker import Tracker
from src.wrapper.utils.feature_exactor import FeatureExtractor
import torch
import numpy as np
import queue
import threading
from typing import Dict, List

import warnings
warnings.filterwarnings('always')

# step说明
# 0 完成检测图片
# 1 开始检测图片
# 2 完成添加图片
# 3 开始添加图片
# 4 完成添加image id
DETECT_IMAGE_COMPLETE = 0
DETECT_IMAGE_START = 1
ADD_IMAGE_COMPLETE = 2
ADD_IMAGE_START = 3
ADD_IMAGE_ID_COMPLETE = 4

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

class DeepSortTracker(object):
    class ImgInfo:
        image_id = 0
        image_shape = np.zeros((0, 0, 0))
        results: Dict[int, List[float]] = {}
        results_update: Dict[int, bool] = {}
        is_used = False
        step = ADD_IMAGE_ID_COMPLETE
        lock = threading.Lock()
    
    def __init__(self, device, weights):
        parse = Parse()
        self.min_confidence = parse.MIN_CONFIDENCE
        self.nms_max_overlap = parse.NMS_MAX_OVERLAP
        self.extractor = FeatureExtractor(weights, device)
        max_cosine_distance = parse.MAX_DIST
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, Parse.NN_BUDGET)
        self.tracker = Tracker(
            metric, max_iou_distance=parse.MAX_IOU_DISTANCE, max_age=parse.MAX_AGE, n_init=parse.N_INIT)
        
        # 图像、结果缓存
        self.max_tracking_length = 10
        self._max_cache = 100
        self._img_infos:Dict[int, DeepSortTracker.ImgInfo] = dict()
        self._img_image_id_fifo:queue.Queue[int] = queue.Queue(maxsize=self._max_cache)

        # 最新检测完成的image_id
        self.latest_detection_completed_image_id = 0
        
        # 打印结果
        self.print_result = False

    def check_image_id_exist(self, image_id) -> bool:
        return image_id in self._img_infos

    def get_statue(self, image_id) -> int:
        if not self.check_image_id_exist(image_id):
            # warnings.warn("该image_id不存在", UserWarning)
            return -1
        return self._img_infos[image_id].step
    
    def add_image_id(self, image_id) -> bool:
        if self.check_image_id_exist(image_id):
            warnings.warn("该image_id已存在", UserWarning)
            return False
        # 清理溢出
        if (len(self._img_infos) >= self._max_cache):
            image_id_rm = self._img_image_id_fifo.get()
            if not self._img_infos[image_id_rm].is_used:
                warnings.warn(f'弹出image_id={image_id_rm}-未被使用', UserWarning)
            with self._img_infos[image_id_rm].lock:
                self._img_infos.pop(image_id_rm)
        self._img_image_id_fifo.put(image_id)
        self._img_infos[image_id] = DeepSortTracker.ImgInfo()
        self._img_infos[image_id].step = ADD_IMAGE_ID_COMPLETE
        return True
    
    # 添加、检测
    def add_image_and_bboxes(self, image_id, image, normalized_bboxes):
        if not self.check_image_id_exist(image_id):
            self.add_image_id(image_id)
        elif self._img_infos[image_id].step != ADD_IMAGE_ID_COMPLETE:
            print(self._img_infos[image_id].step)
            warnings.warn("重复添加", UserWarning)
            return False
        self._img_infos[image_id].step = ADD_IMAGE_COMPLETE
        self._img_infos[image_id].image_shape = image.shape
        result = None
        self._img_infos[image_id].step = DETECT_IMAGE_START
        results = []
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
        results = self.update(xywh, confss, image)
        
        # 将全部id置为未更新
        for id in self._img_infos[image_id].results_update:
            self._img_infos[image_id].results_update[id] = False
        # 更新id的result
        for result in results:
            id = result[4]
            # 新的id
            if id not in self._img_infos[image_id].results_update:
                self._img_infos[image_id].results[id] = []
            # 判断结果数量
            if len(self._img_infos[image_id].results[id]) > self.max_tracking_length:
                self._img_infos[image_id].results[id] = self._img_infos[image_id].results[id][1:]
            # 添加结果
            self._img_infos[image_id].results[id].append([
                float(result[0]) / image.shape[1],
                float(result[1]) / image.shape[0],
                float(result[2]) / image.shape[1],
                float(result[3]) / image.shape[0],
            ])
            # 更新id状态
            self._img_infos[image_id].results_update[id] = True
        # 删除未更新的id
        deleteId = []
        for id in self._img_infos[image_id].results_update:
            update = self._img_infos[image_id].results_update[id]
            if not update:
                deleteId.append(id)
        for id in deleteId:
            del self._img_infos[image_id].results[id]
            del self._img_infos[image_id].results_update[id]
        self._img_infos[image_id].step = DETECT_IMAGE_COMPLETE
        self.latest_detection_completed_image_id = image_id
        return True

    def get_result_by_image_id(self, image_id) -> Dict[int, List[float]]:
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image_id不存在", UserWarning)
            return None
        if self._img_infos[image_id].step != DETECT_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return False
        with self._img_infos[image_id].lock:
            self._img_infos[image_id].is_used = True
            if self.print_result and len(self._img_infos[image_id].results):
                print(self._img_infos[image_id].results)
        return self._img_infos[image_id].results

    def update(self, bboxes_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        im_crops = []
        features = np.array([])
        for bbox_xywh in bboxes_xywh:
            x, y, w, h = bbox_xywh
            x1 = max(int(x - w / 2), 0)
            x2 = min(int(x + w / 2), self.width - 1)
            y1 = max(int(y - h / 2), 0)
            y2 = min(int(y + h / 2), self.height - 1)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        if isinstance(bboxes_xywh, np.ndarray):
            bboxes_tlwh = bboxes_xywh.copy()
        elif isinstance(bboxes_xywh, torch.Tensor):
            bboxes_tlwh = bboxes_xywh.clone()
        bboxes_tlwh[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2.
        bboxes_tlwh[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2.
        detections = [Detection(bboxes_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bboxes_tlwh = track.to_tlwh()
            x, y, w, h = bboxes_tlwh
            x1 = max(int(x), 0)
            x2 = min(int(x+w), self.width - 1)
            y1 = max(int(y), 0)
            y2 = min(int(y+h), self.height - 1)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs
