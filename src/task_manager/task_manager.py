from src.utils import singleton
from typing import Dict, Tuple, List
from queue import Queue
from src.grpc.clients.image_harmony.image_harmony_client import ImageHarmonyClient
from src.grpc.clients.target_detection.target_detection_client import TargetDetectionClient
from src.config.config import Config
import _thread
from wrapper.deepsort_tracker import DeepSortTracker
import traceback

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
    def __init__(self, taskId: int):
        config = Config()
        
        self.id: int = taskId
        self.stop: bool = True
        
        self.image_harmony_address: List[str, str] = []
        self.image_harmony_client: ImageHarmonyClient = None
        self.loader_args_hash: int = 0  # image harmony中加载器的hash值
        
        self.target_detection_address: List[str, str] = []
        self.target_detection_client: TargetDetectionClient = None
        self.target_label: str = 'person'  # 默认跟踪person

        self.weights_folder = config.weights_folder
        self.weight: str = 'ckpt.t7'
        self.device: str = ''
        self.max_tracking_length: int = 10
        self.image_id_queue: Queue[int] = Queue()
        self.tracker: DeepSortTracker = None
    
    def set_pre_service(self, pre_service_name: str, pre_service_ip: str, pre_service_port: str, args: Dict[str, str]):
        if 'image harmony' == pre_service_name:
            self.image_harmony_address = [pre_service_ip, pre_service_port]
            self.image_harmony_client = ImageHarmonyClient(pre_service_ip, pre_service_port)
            assert 'LoaderArgsHash' in args, 'arg: [LoaderArgsHash] not set'
            self.loader_args_hash = int(args['LoaderArgsHash'])
        if 'target detection' == pre_service_name:
            self.target_detection_address = [pre_service_ip, pre_service_port]
            self.target_detection_client = TargetDetectionClient(pre_service_ip, pre_service_port, self.id)
    
    def set_cur_service(self, args: Dict[str, str]):
        # TODO weights以后也通过配置文件传
        if 'Device' in args:
            self.device = args['Device']
        if 'Weight' in args:
            self.weight = args['Weight']
        if 'TargetLabel' in args:
            self.target_label = args['TargetLabel']
        if 'MaxTrackingLength' in args:
            self.max_tracking_length = int(args['MaxTrackingLength'])
    
    def check(self) -> Tuple[bool, str]:
        try:
            assert self.target_detection_client, 'Error: target_detection_client not set.'
            assert self.image_harmony_client,    'Error: image_harmony_client not set.'
            assert self.loader_args_hash,        'Error: loader_args_hash not set.'
            assert self.weight,                  'Error: weight not set.'
            assert self.device,                  'Error: device not set.'
            assert self.target_label,            'Error: target_label not set.'
            # assert self.tracker,                 'Error: tracker not set.'
        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
            return False, error_info
        return True, 'OK'
    
    def start(self):
        self.image_harmony_client.set_loader_args_hash(self.loader_args_hash)
        self.tracker = DeepSortTracker(
            device=self.device,
            weights=f'{self.weights_folder}/{self.weight}'
        )
        self.tracker.max_tracking_length = self.max_tracking_length
        self.target_detection_client.set_track_target_label(self.target_label)
        self.stop = False
        # _thread.start_new_thread(self.progress, ())
        # TODO 临时版本
        _thread.start_new_thread(self.track_by_image_id, ())
    
    def track_by_image_id(self):
        assert self.tracker, 'tracker is not set\n'
        assert self.image_harmony_client, 'image harmony client is not set\n'
        while not self.stop:
            image_id_in_queue = self.image_id_queue.get()
            width, height = self.image_harmony_client.get_image_size_by_image_id(image_id_in_queue)
            if 0 == width or 0 == height:
                continue
            
            new_width, new_height = calculate_scaled_size(width, height)
            image_id, image = self.image_harmony_client.get_image_by_image_id(image_id_in_queue, new_width, new_height)
            if 0 == image_id:
                continue
            bboxs = self.target_detection_client.get_result_by_image_id(image_id)
            if not self.tracker.add_image_and_bboxes(image_id, image, bboxs):
                continue
            result = self.tracker.get_result_by_uid(image_id)
            print(result)
                                                                                                                                           
@singleton
class TaskManager:
    def __init__(self):
        self.tasks_queue: Queue[TaskInfo] = Queue(maxsize=20)
        self.incomplete_tasks: Dict[int, TaskInfo] = {}
        self.tasks: Dict[int, TaskInfo] = {}

    def listening(self):
        def wait_for_task():
            while True:
                task = self.tasks_queue.get()
                task.start()

        _thread.start_new_thread(wait_for_task, ())
