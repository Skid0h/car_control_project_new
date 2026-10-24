from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('car_jetson'))
    control = Node(package='car_jetson', executable='control', output='screen',
                   parameters=[LaunchConfiguration('params'),
                               {'dry_run': ParameterValue(LaunchConfiguration('dry_run'), value_type=bool),
                                'serial_port': ParameterValue(LaunchConfiguration('serial_port'), value_type=str),
                                'project_config': ParameterValue(LaunchConfiguration('project_config'), value_type=str)}])
    vision = Node(package='car_jetson', executable='vision', output='screen',
                  parameters=[LaunchConfiguration('params'),
                              {'model_path': ParameterValue(LaunchConfiguration('model_path'), value_type=str),
                               'project_config': ParameterValue(LaunchConfiguration('project_config'), value_type=str),
                               'web_enabled': ParameterValue(LaunchConfiguration('web_enabled'), value_type=bool)}])
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=str(share / 'config' / 'jetson.yaml')),
        DeclareLaunchArgument('project_config', default_value=str(share / 'config' / 'project.jsonc')),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('serial_port', default_value=''),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('web_enabled', default_value='false'),
        RegisterEventHandler(OnProcessExit(target_action=control,
                             on_exit=[EmitEvent(event=Shutdown(reason='Control exited'))])),
        RegisterEventHandler(OnProcessExit(target_action=vision,
                             on_exit=[EmitEvent(event=Shutdown(reason='Vision exited'))])),
        control, vision,
    ])
