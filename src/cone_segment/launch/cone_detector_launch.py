"""
cone_detector_launch.py
=======================
ros2 launch cone_segment cone_detector_launch.py
ros2 launch cone_segment cone_detector_launch.py camera_index:=1 model_path:=/path/to/best.pt

[추가 파라미터]
  max_disappeared : 소실 허용 프레임 수 (기본 45 = 30fps 기준 1.5초)
  max_cost        : Hungarian 매칭 거부 임계값 (기본 220.0)
                    값이 클수록 멀리 있는 쌍도 같은 ID로 매칭 (느린 UGV에 유리)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    args = [
        DeclareLaunchArgument('camera_index',    default_value='0'),
        DeclareLaunchArgument('model_path',      default_value='/home/woohyuck/capstone_ws/src/cone_segment/model/best7.pt'),
        DeclareLaunchArgument('hfov_deg',        default_value='46.0'),
        DeclareLaunchArgument('conf_thresh',     default_value='0.85'),
        DeclareLaunchArgument('show_image',      default_value='true'),
        DeclareLaunchArgument('img_width',       default_value='640'),
        DeclareLaunchArgument('img_height',      default_value='480'),
        # ── 추적기 파라미터 ──────────────────────────────────────────────────
        DeclareLaunchArgument('max_disappeared', default_value='45'),
        DeclareLaunchArgument('max_cost',        default_value='150.0'),
    ]

    torch_python = '/home/woohyuck/miniconda3/envs/torch/bin/python'
    node_script  = '/home/woohyuck/capstone_ws/src/cone_segment/cone_segment/cone_detector_node.py'

    cone_detector = ExecuteProcess(
        cmd=[
            torch_python, node_script,
            '--ros-args',
            '-r', '__node:=cone_detector_node',
            '-p', ['camera_index:=',    LaunchConfiguration('camera_index')],
            '-p', ['model_path:=',      LaunchConfiguration('model_path')],
            '-p', ['hfov_deg:=',        LaunchConfiguration('hfov_deg')],
            '-p', ['conf_thresh:=',     LaunchConfiguration('conf_thresh')],
            '-p', ['show_image:=',      LaunchConfiguration('show_image')],
            '-p', ['img_width:=',       LaunchConfiguration('img_width')],
            '-p', ['img_height:=',      LaunchConfiguration('img_height')],
            '-p', ['max_disappeared:=', LaunchConfiguration('max_disappeared')],
            '-p', ['max_cost:=',        LaunchConfiguration('max_cost')],
        ],
        output='screen',
    )

    return LaunchDescription(args + [cone_detector])
