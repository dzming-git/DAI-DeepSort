from concurrent import futures
import time
import grpc
import traceback
from generated.protos.service_coordinator import service_coordinator_pb2, service_coordinator_pb2_grpc
from typing import Dict
from src.task_manager.task_manager import TaskManager, TaskInfo

VALID_PRE_SERVICE = ['image harmony', 'target detection']

class ServiceCoordinatorServer(service_coordinator_pb2_grpc.CommunicateServicer):
    def informPreviousServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            task_info: TaskInfo = None

            if task_id in task_manager.tasks:
                # task_id在task_manager.tasks存在
                # 修改已有任务的参数
                task_info = task_manager.tasks[task_id]
            else:
                # 新的task_id
                task_info = TaskInfo(request.taskId)
                task_manager.incomplete_tasks[task_id] = task_info
                task_manager.tasks[task_id] = task_info
            assert request.preServiceName in VALID_PRE_SERVICE, 'invalid pre service\n'
            args = {}
            for arg in request.args:
                args[arg.key] = arg.value
            task_info.set_pre_service(
                pre_service_name=request.preServiceName,
                pre_service_ip=request.preServiceIp,
                pre_service_port=request.preServicePort,
                args=args)
            if task_id in task_manager.incomplete_tasks:
                ok, msg = task_info.check()
                if ok:
                    task_manager.tasks_queue.put(task_info)
                    task_manager.incomplete_tasks.pop(task_id)

        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def informCurrentServiceInfo(self, request, context):
        print(request)
        response_code = 200
        response_message = ''
        # TODO: 目前只支持weight、ConnectID
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            task_info: TaskInfo = None

            if task_id in task_manager.tasks:
                # task_id在task_manager.tasks存在
                # 修改已有任务的参数
                task_info = task_manager.tasks[task_id]
            else:
                # 新的task id
                task_info = TaskInfo(request.taskId)
                task_manager.incomplete_tasks[task_id] = task_info
                task_manager.tasks[task_id] = task_info
            
            if task_id in task_manager.incomplete_tasks:
                # task_id在task_manager.incomplete_tasks存在
                # 修改已有任务的参数
                task_info = task_manager.incomplete_tasks[task_id]

            args = {}
            for arg in request.args:
                args[arg.key] = arg.value
            task_info.set_cur_service(args)
            
            if task_id in task_manager.incomplete_tasks:
                ok, msg = task_info.check()
                if ok:
                    task_manager.tasks_queue.put(task_info)
                    task_manager.incomplete_tasks.pop(task_id)
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()

        response = service_coordinator_pb2.InformPreviousServiceInfoResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response

    def start(self, request, context):
        response_code = 200
        response_message = ''
        try:
            task_manager = TaskManager()
            task_id = request.taskId
            assert task_id in task_manager.tasks, 'ERROR: The task ID does not exist.\n'
            tracker = task_manager.tasks[task_id].tracker
        
        except Exception as e:
            response_code = 400
            response_message += traceback.format_exc()
        response = service_coordinator_pb2.StartResponse()
        response.response.code = response_code
        response.response.message = response_message
        return response
    
    def join_in_server(self, server):
        service_coordinator_pb2_grpc.add_CommunicateServicer_to_server(self, server.server)
