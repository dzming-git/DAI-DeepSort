from deep_sort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.sort.preprocessing import non_max_suppression
from deep_sort.deep_sort.sort.detection import Detection
from deep_sort.deep_sort.sort.tracker import Tracker
from src.wrapper.utils.feature_exactor import FeatureExtractor
import torch
import numpy as np
import queue
import threading
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('always')

# step说明
DETECT_IMAGE_COMPLETE: int = 0
DETECT_IMAGE_START: int = 1
ADD_IMAGE_COMPLETE: int = 2
ADD_IMAGE_START: int = 3
ADD_IMAGE_ID_COMPLETE: int = 4

class DeepSort(object):
    class DeepSortBuilder:
        def __init__(self) -> None:
            self.device_str: str = 'cpu'  # cuda:0 cuda:1
            self.weights: str = 'weights/ckpt.t7'
            self.max_dist: float = 0.2
            self.min_confidence: float = 0.3
            self.nms_max_overlap: float = 0.5
            self.nn_budget: Optional[int] = 100
            self.max_iou_distance: float = 0.7
            self.max_age: int = 70
            self.n_init: int = 3
            self.max_tracking_length: int = 10
            self.max_cache: int = 100
            self.print_result: bool = False
        
        def build(self) -> 'DeepSort':
            if not torch.cuda.is_available():
                if self.device_str != 'cpu':
                    warnings.warn("cuda is not available", UserWarning)
                self.device_str = 'cpu'
            return DeepSort(self)
    
    class ImgInfo:
        def __init__(self) -> None:
            self.image_id: int = 0
            self.image_shape: np.ndarray = np.zeros((0, 0, 0))
            self.is_used: bool = False
            self.step: int = ADD_IMAGE_ID_COMPLETE
            self.lock: threading.Lock = threading.Lock()
    
    def __init__(self, builder: DeepSortBuilder) -> None:
        self.__device_str: str = builder.device_str
        self.__device: torch.device = torch.device(builder.device_str)
        self.__weights: str = builder.weights
        self.__min_confidence: float = builder.min_confidence
        self.__nms_max_overlap: float = builder.nms_max_overlap
        self.__max_tracking_length: int = builder.max_tracking_length
        self.__max_cache: int = builder.max_cache
        self.__print_result: bool = builder.print_result

        self.__extractor: FeatureExtractor = FeatureExtractor(self.__weights, self.__device)
        metric: NearestNeighborDistanceMetric = NearestNeighborDistanceMetric("cosine", builder.max_dist, builder.nn_budget)
        self.__tracker: Tracker = Tracker(metric, max_iou_distance=builder.max_iou_distance, max_age=builder.max_age, n_init=builder.n_init)

        self.__img_infos: Dict[int, DeepSort.ImgInfo] = dict()
        self.__img_image_id_fifo: queue.Queue[int] = queue.Queue(maxsize=self.__max_cache)
        
        self.__results: Dict[int, List[float]] = {}
        self.__results_update_flag: Dict[int, bool] = {}

        self.latest_detection_completed_image_id: int = 0

    def check_image_id_exist(self, image_id: int) -> bool:
        return image_id in self.__img_infos

    def get_status(self, image_id: int) -> int:
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image_id不存在", UserWarning)
            return -1
        return self.__img_infos[image_id].step
    
    def add_image_id(self, image_id: int) -> bool:
        if self.check_image_id_exist(image_id):
            warnings.warn("该image_id已存在", UserWarning)
            return False
        if len(self.__img_infos) >= self.__max_cache:
            image_id_rm: int = self.__img_image_id_fifo.get()
            if not self.__img_infos[image_id_rm].is_used:
                warnings.warn(f'弹出image_id={image_id_rm}-未被使用', UserWarning)
            with self.__img_infos[image_id_rm].lock:
                self.__img_infos.pop(image_id_rm)
        self.__img_image_id_fifo.put(image_id)
        self.__img_infos[image_id] = DeepSort.ImgInfo()
        self.__img_infos[image_id].step = ADD_IMAGE_ID_COMPLETE
        return True

    def add_image_and_bboxes(self, image_id: int, image: np.ndarray, normalized_bboxes: List[List[float]]) -> bool:
        if not self.check_image_id_exist(image_id):
            self.add_image_id(image_id)
        elif self.__img_infos[image_id].step != ADD_IMAGE_ID_COMPLETE:
            print(self.__img_infos[image_id].step)
            warnings.warn("重复添加", UserWarning)
            return False
        self.__img_infos[image_id].step = ADD_IMAGE_COMPLETE
        self.__img_infos[image_id].image_shape = image.shape
        result = None
        self.__img_infos[image_id].step = DETECT_IMAGE_START
        results = []
        xywh = []
        for normalized_bbox in normalized_bboxes:
            bbox = [
                normalized_bbox[0] * image.shape[1],  # 左上角 x 坐标
                normalized_bbox[1] * image.shape[0],  # 左上角 y 坐标
                normalized_bbox[2] * image.shape[1],  # 右下角 x 坐标
                normalized_bbox[3] * image.shape[0]   # 右下角 y 坐标
            ]
            bbox_left = min([bbox[0], bbox[2]])
            bbox_top = min([bbox[1], bbox[3]])
            bbox_w = abs(bbox[0] - bbox[2])
            bbox_h = abs(bbox[1] - bbox[3])
            x_c = (bbox_left + bbox_w / 2)
            y_c = (bbox_top + bbox_h / 2)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh.append(xywh_obj)
        xywh = torch.Tensor(xywh)
        confss = torch.from_numpy(np.ones(shape=(len(normalized_bboxes))))
        results = self.__update(xywh, confss, image)
        
        # 将全部id置为未更新
        for id in self.__results_update_flag:
            self.__results_update_flag[id] = False
        # 更新id的result
        for result in results:
            id = result[4]
            # 新的id
            if id not in self.__results_update_flag:
                self.__results[id] = []
            # 判断结果数量
            if len(self.__results[id]) > self.__max_tracking_length:
                self.__results[id] = self.__results[id][1:]
            # 添加结果
            self.__results[id].append([
                float(result[0]) / image.shape[1],
                float(result[1]) / image.shape[0],
                float(result[2]) / image.shape[1],
                float(result[3]) / image.shape[0],
            ])
            # 更新id状态
            self.__results_update_flag[id] = True
        # 删除未更新的id
        deleteId = []
        for id in self.__results_update_flag:
            update = self.__results_update_flag[id]
            if not update:
                deleteId.append(id)
        for id in deleteId:
            del self.__results[id]
            del self.__results_update_flag[id]
        self.__img_infos[image_id].step = DETECT_IMAGE_COMPLETE
        self.latest_detection_completed_image_id = image_id
        return True

    def get_result_by_image_id(self, image_id: int) -> Dict[int, List[float]]:
        if not self.check_image_id_exist(image_id):
            warnings.warn("该image_id不存在", UserWarning)
            return {}
        if self.__img_infos[image_id].step != DETECT_IMAGE_COMPLETE:
            warnings.warn("图片未添加完成", UserWarning)
            return {}
        
        with self.__img_infos[image_id].lock:
            self.__img_infos[image_id].is_used = True
            if self.__print_result and len(self.__results):
                print(self.__results)
        return self.__results

    def __update(self, bboxes_xywh: torch.Tensor, confidences: torch.Tensor, ori_img: np.ndarray) -> List[np.ndarray]:
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
            features = self.__extractor(im_crops)
        if isinstance(bboxes_xywh, np.ndarray):
            bboxes_tlwh = bboxes_xywh.copy()
        elif isinstance(bboxes_xywh, torch.Tensor):
            bboxes_tlwh = bboxes_xywh.clone()
        bboxes_tlwh[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2.
        bboxes_tlwh[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2.
        detections = [Detection(bboxes_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.__min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.__nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.__tracker.predict()
        self.__tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.__tracker.tracks:
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