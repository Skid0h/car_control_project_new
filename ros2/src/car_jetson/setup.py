from glob import glob
from setuptools import setup
setup(
    name='car_jetson', version='0.1.0', packages=['car_jetson'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/car_jetson']),
        ('share/car_jetson', ['package.xml']),
        ('share/car_jetson/launch', glob('launch/*.launch.py')),
        ('share/car_jetson/config', glob('config/*')),
    ],
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Car project maintainers', maintainer_email='maintainer@example.com',
    description='Native ROS 2 Jetson car control', license='LicenseRef-Proprietary',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'control = car_jetson.control_node:main',
        'vision = car_jetson.vision_node:main',
        'preflight = car_jetson.preflight:main',
    ]},
)
