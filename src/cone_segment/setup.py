from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cone_segment'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='YOLO Segmentation 기반 꼬깔 검출 ROS2 패키지',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_detector_node    = cone_segment.cone_detector_node:main',
            'speed_controller_node = cone_segment.speed_controller_node:main',
            'test_sub              = cone_segment.test_sub:main',
        ],
    },
)