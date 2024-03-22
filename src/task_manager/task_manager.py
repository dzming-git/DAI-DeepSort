from src.utils import singleton
from typing import Dict, Tuple, List
import queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.grpc.clients.target_detection.target_detection_client import TargetDetectionClient
from src.config.config import Config
from src.wrapper.deepsort_tracker import DeepSort
import traceback
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_scaled_size(width: int, height: int) -> Tuple[int, int]:
    max_width = 640
    max_height = 480

    # 如果原始尺寸小于规定尺寸，直接返回原始尺寸
    if width <= max_width and height <= max_height:
        return width, height

    # 计算宽度和高度的缩放比例
    width_ratio = max_width / width
    height_ratio = max_height / height

    # 选择较小的缩放比例，确保图像适应最大尺寸限制
    scale_ratio = min(width_ratio, height_ratio)

    # 计算缩放后的宽度和高度
    scaled_width = int(width * scale_ratio)
    scaled_height = int(height * scale_ratio)

    return scaled_width, scaled_height

class TaskInfo:
    def __init__(self, task_id: int):
        config = Config()
        
        self.id: int = task_id
        
        self.image_harmony_address: List[str, str] = []
        self.image_harmony_client: ImageHarmonyClient = None
        self.loader_args_hash: int = 0  # image harmony中加载器的hash值
        
        self.target_detection_address: List[str, str] = []
        self.target_detection_client: TargetDetectionClient = None
        self.target_label: str = 'person'  # 默认跟踪person

        self.weights_folder = config.weights_folder
        self.weight: str = 'ckpt.t7'
        self.device_str: str = ''
        self.max_tracking_length: int = 10
        self.image_id_queue: queue.Queue[int] = queue.Queue()
        self.tracker: DeepSort = None
        self.stop_event = threading.Event()
        self.track_thread = None  # 用于跟踪线程的引用
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str]):
        if 'image harmony gRPC' == pre_service_name:
            self.image_harmony_address = [pre_service_ip, pre_service_port]
            self.image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            if 'LoaderArgsHash' not in args:
                raise ValueError('Argument "LoaderArgsHash" is required but not set.')
            self.loader_args_hash = int(args['LoaderArgsHash'])
        if 'target detection' == pre_service_name:
            self.target_detection_address = [pre_service_ip, pre_service_port]
            self.target_detection_client = TargetDetectionClient(pre_service_ip, pre_service_port, self.id)
    
    def set_cur_service(self, args: Dict[str, str]):
        # TODO weights以后也通过配置文件传
        if 'Device' in args:
            self.device_str = args['Device']
        if 'Weight' in args:
            self.weight = args['Weight']
        if 'TargetLabel' in args:
            self.target_label = args['TargetLabel']
        if 'MaxTrackingLength' in args:
            self.max_tracking_length = int(args['MaxTrackingLength'])
    
    def check(self) -> Tuple[bool, str]:
        if not self.target_detection_client:
            raise ValueError('Error: target_detection_client not set.')
        if not self.image_harmony_client:
            raise ValueError('Error: image_harmony_client not set.')
        if not self.loader_args_hash:
            raise ValueError('Error: loader_args_hash not set.')
        if not self.weight:
            raise ValueError('Error: weight not set.')
        if not self.device_str:
            raise ValueError('Error: device not set.')
        if not self.target_label:
            raise ValueError('Error: target_label not set.')
    
    def start(self) -> None:
        self.check()
        self.image_harmony_client.connect_image_loader(self.loader_args_hash)
        builder = DeepSort.DeepSortBuilder()
        builder.device_str = self.device_str
        builder.weights = f'{self.weights_folder}/{self.weight}'
        builder.max_tracking_length = self.max_tracking_length
        self.tracker = builder.build()
        self.target_detection_client.filter.clear()
        target_label_id = self.target_detection_client.query_label_id(self.target_label)
        self.target_detection_client.filter.add(target_label_id)
        self.stop_event.clear()  # 确保开始时事件是清除状态
        self.track_thread = threading.Thread(target=self.track_by_image_id)
        self.track_thread.start()
    
    def track_by_image_id(self):
        while not self.stop_event.is_set():  # 使用事件来检查停止条件
            # image_id_in_queue = self.image_id_queue.get()
            try:
            # 尝试从队列中获取image_id，设置超时时间为1秒
                image_id_in_queue = self.image_id_queue.get(timeout=1)
            except queue.Empty:
                # 如果在超时时间内没有获取到新的image_id，则继续循环，此时可以检查停止事件
                continue
            try:
                width, height = self.image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
                if 0 == width or 0 == height:
                    continue
                
                new_width, new_height = calculate_scaled_size(width, height)
                if self.stop_event.is_set():  # 在可能的长时间操作之前再次检查
                    break
                image_id, image = self.image_harmony_client.get_image_by_image_id(image_id_in_queue, new_width, new_height)
                if 0 == image_id:
                    continue
                if self.stop_event.is_set():  # 在可能的长时间操作之前再次检查
                    break
                results = self.target_detection_client.get_result_by_image_id(image_id)
                bboxs: List[List[float]] = []
                for result in results:
                    bboxs.append(
                        [
                            result.x1,
                            result.y1,
                            result.x2,
                            result.y2
                        ]
                    )
                if not self.tracker.add_image_and_bboxes(image_id, image, bboxs):
                    continue
            except Exception as e:
                logging.error(e)

            # result = self.tracker.get_result_by_uid(image_id)
            # print(result)
    
    def stop(self) -> None:
        self.stop_event.set()  # 设置事件，通知线程停止
        if self.track_thread:
            self.track_thread.join()  # 等待线程结束
        self.image_harmony_client.disconnect_image_loader()
        if self.tracker:
            del self.tracker  # 释放资源
        self.tracker = None
                                                                                                                                           
@singleton
class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, TaskInfo] = {}
        self.__lock = threading.Lock()
    
    def stop_task(self, task_id: int):
        with self.__lock:
            if task_id in self.tasks:
                self.tasks[task_id].stop()
                del self.tasks[task_id]
