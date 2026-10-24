from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from .config_load import Config


def configuration(node):
    default = str(Path(get_package_share_directory('car_jetson')) / 'config' / 'project.jsonc')
    filename = node.declare_parameter('project_config', default).value
    return Config(filename)


def value(node, name, default):
    return node.declare_parameter(name, default).value
