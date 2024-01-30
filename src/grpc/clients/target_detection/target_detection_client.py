from generated.protos.target_detection import target_detection_pb2, target_detection_pb2_grpc
import grpc
import cv2
from typing import Dict, Tuple, List
import numpy as np

class TargetDetectionClient:
    def __init__(self, ip:str, port: str, taskId: int):
        self.conn = grpc.insecure_channel(f'{ip}:{port}')
        self.client = target_detection_pb2_grpc.CommunicateStub(channel=self.conn)
        self.task_id: int = taskId
        self.track_target_label: str = ''
        self.target_lable_id = -1
    
    def set_task_id(self, task_id: int):
        self.task_id = task_id
    
    def set_track_target_label(self, track_target_label: str):
        self.track_target_label = track_target_label
        ok = self.get_target_label_id()
        return ok
    
    def get_target_label_id(self):
        # 只需要筛选出person的id
        request = target_detection_pb2.GetResultMappingTableRequest()
        request.taskId = self.task_id
        response = self.client.getResultMappingTable(request)
        if 200 != response.response.code:
            print(f'{response.response.code}: {response.response.message}')
            return False
        for i, label in enumerate(response.labels):
            if self.track_target_label == label:
                self.target_lable_id = i
                return True
        return False
    
    def get_result_by_image_id(self, image_id: int) -> List[List[float]]:
        request = target_detection_pb2.GetResultIndexByImageIdRequest()
        request.taskId = self.task_id
        request.imageId = image_id
        request.wait = True
        response = self.client.getResultIndexByImageId(request)
        if 200 != response.response.code:
            print(f'{response.response.code}: {response.response.message}')
            return []
        results: List[List[float]] = []
        for result in response.results:
            if result.labelId != self.target_lable_id:
                continue
            confidence = result.confidence
            x1 = result.x1
            y1 = result.y1
            x2 = result.x2
            y2 = result.y2
            results.append([x1, y1, x2, y2])
        return results
