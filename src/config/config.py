from src.utils import singleton
import yaml
from typing import List

CONFIG_PATH = './.config.yml'

class WeightInfo:
    file: str
    labels: List[str]

@singleton
class Config:
    def __init__(self):
        # service
        self.service_name: str = ''
        self.service_port: int = -1
        self.service_tags: List[str] = list()
        self.weights_folder: str = ''

        #consul
        self.consul_ip: str = ''
        self.consul_port: int = -1
        
        self.load_config()

    def load_config(self):
        with open(CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        service_data = config_data.get('service', {})
        self.service_name = service_data.get('name', '')
        self.service_port = service_data.get('port', -1)
        self.service_tags = service_data.get('tags', [])
        self.weights_folder = service_data.get('weights_folder', './weights')
        consul_data = config_data.get('consul', {})
        self.consul_ip = consul_data.get('ip', '')
        self.consul_port = consul_data.get('port', -1)
