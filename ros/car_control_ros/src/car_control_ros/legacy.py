"""Load existing configuration/detector without importing hardware server.py."""
import importlib.util
from pathlib import Path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration():
    import rospy
    import rospkg
    package = Path(rospkg.RosPack().get_path('car_control_ros')).resolve()
    default = package / 'legacy'
    if not default.exists():
        default = package.parent.parent / 'Jetson Xavier'
    root = Path(rospy.get_param('~legacy_dir', str(default))).expanduser().resolve()
    cls = load_module(root / 'Code' / 'Config_load.py', 'legacy_config').Config
    return root, cls(rospy.get_param('~project_config', str(root / 'config.jsonc')))
