from src.task_manager.task_manager import TaskManager
from src.config.config import Config
from src.grpc.servers.grpc_server_builder import GRPCServerBuilder
import time
from src.grpc.servers.service_coordinator.service_coordinator_server import ServiceCoordinatorServer
from src.grpc.servers.target_tracking.target_tracking_server import TargetTrackingServer

service_coordinator_server = ServiceCoordinatorServer()
target_tracking_server = TargetTrackingServer()
config = Config()
gRPCServer = None

def connect_consul():
    from src.consul import ConsulClient
    from src.consul import ServiceInfo
    from src.utils import get_local_ip_address

    consul_client = ConsulClient()
    consul_client.consul_ip = config.consul_ip
    consul_client.consul_port = config.consul_port
    host = get_local_ip_address()
    service_name = config.service_name
    service_port = config.service_port
    service_info = ServiceInfo()
    service_info.service_ip = host
    service_info.service_port = service_port
    service_info.service_name = service_name
    service_info.service_id = f'{service_name}-{host}-{service_port}'
    service_info.service_tags = config.service_tags
    consul_client.register_service(service_info)

def gRPC_server_start():
    global gRPCServer
    gRPCServerBuilder = GRPCServerBuilder()
    gRPCServer = gRPCServerBuilder.build()
    service_coordinator_server.join_in_server(gRPCServer)
    target_tracking_server.join_in_server(gRPCServer)
    gRPCServer.start()

if __name__ == '__main__':
    connect_consul()
    
    gRPC_server_start()

    while True:
        time.sleep(60 * 60 * 24)
