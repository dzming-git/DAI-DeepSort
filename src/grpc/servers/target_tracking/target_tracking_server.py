from generated.protos.target_tracking import target_tracking_pb2, target_tracking_pb2_grpc
import time
import traceback
from src.task_manager.task_manager import TaskManager
from src.config.config import Config

task_manager = TaskManager()
config = Config()

class TargetTrackingServer(target_tracking_pb2_grpc.CommunicateServicer):
    def getResultByImageId(self, request, context):
        response_code = 200
        response_message = ''
        results_dict = {}
        try:
            task_id = request.taskId
            image_id = request.imageId
            wait = request.wait
            only_the_latest = request.onlyTheLatest
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            task = task_manager.tasks[task_id]
            tracker = task.tracker
            # TODO 初步使用同步模式
            image_id_exist = tracker.check_image_id_exist(image_id)
            if not image_id_exist and wait:
                task_manager.tasks[task_id].image_id_queue.put(image_id)
            # 设置超时时间为 1 秒
            timeout = 1
            start_time = time.time()

            # 等待添加完成
            while not tracker.check_image_id_exist(image_id):
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("添加图片超时")
                time.sleep(0.01)
            
            # 等待检测完成
            while tracker.get_status(image_id) != 0:
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("检测图片超时")
                time.sleep(0.01)
            
            results_dict = tracker.get_result_by_image_id(image_id)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = target_tracking_pb2.GetResultByImageIdResponse()
        response.response.code = response_code
        response.response.message = response_message
        for id in results_dict:
            results = results_dict[id]
            result_response = target_tracking_pb2.Result()
            result_response.id = id
            result_response.label = task.target_label
            if only_the_latest and results:
                bbox = result_response.bboxs.add()
                x1, y1, x2, y2 = results[-1]
                bbox.x1 = x1
                bbox.y1 = y1
                bbox.x2 = x2
                bbox.y2 = y2
            else:
                for result in results:
                    bbox = result_response.bboxs.add()
                    x1, y1, x2, y2 = result
                    bbox.x1 = x1
                    bbox.y1 = y1
                    bbox.x2 = x2
                    bbox.y2 = y2
            response.results.append(result_response)
        return response
    
    def join_in_server(self, server):
        target_tracking_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
