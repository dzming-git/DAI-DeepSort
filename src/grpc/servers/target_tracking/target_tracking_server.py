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
        results = []
        try:
            task_id = request.taskId
            image_id = request.imageId
            wait = request.wait
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            task = task_manager.tasks[task_id]
            tracker = task.tracker
            # TODO 初步使用同步模式
            image_id_exist = tracker.check_uid_exist(image_id)
            if not image_id_exist and wait:
                task_manager.tasks[task_id].image_id_queue.put(image_id)
            # 设置超时时间为 1 秒
            timeout = 1
            start_time = time.time()

            # 等待检测完成
            while tracker.get_statue(image_id) != 0:
                # 检查是否超过了超时时间
                if time.time() - start_time > timeout:
                    raise TimeoutError("等待超时")
                
                time.sleep(0.01)
            
            results = tracker.get_result_by_uid(image_id)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = target_tracking_pb2.GetResultByImageIdResponse()
        response.response.code = response_code
        response.response.message = response_message
        for result in results:
            result_response = target_tracking_pb2.Result()
            x1, y1, x2, y2, id = result
            result_response.id = id
            result_response.x1 = x1
            result_response.y1 = y1
            result_response.x2 = x2
            result_response.y2 = y2
            response.results.append(result_response)
        return response
    
    def join_in_server(self, server):
        target_tracking_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
