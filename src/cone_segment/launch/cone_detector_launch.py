"""
cone_detector_launch.py
=======================
꼬깔 인식 + 속도 제어 통합 실행.

실행 노드:
    1. cone_detector_node   : YOLO segmentation 기반 꼬깔 인식 → /cone_pairs
    2. speed_controller_node: /cone_pairs 구독 → PX4 velocity 제어

사용법:
    ros2 launch cone_segment cone_detector_launch.py
    ros2 launch cone_segment cone_detector_launch.py u_max:=1.5 speed_k:=0.8

파라미터:
  [인식]
    camera_index    : 카메라 인덱스 (기본 0)
    model_path      : YOLO 모델 경로
    hfov_deg        : 카메라 수평 FOV [deg] (기본 46.0)
    conf_thresh     : YOLO confidence 임계값 (기본 0.85)
    cone_real_height: 실제 꼬깔 높이 [m] (기본 0.23)
    max_disappeared : 소실 허용 프레임 수 (기본 45)
    max_cost        : Hungarian 매칭 거부 임계값 (기본 150.0)
  [속도 제어]
    u_min           : 최소 전진 속도 [m/s] (기본 1.5)
    u_max           : 최대 전진 속도 [m/s] (기본 2.0)
    speed_k         : distance → speed 비례계수 (기본 1.0)
    use_distance    : 거리 기반 가변 속도 사용 여부 (기본 true)
    cmd_timeout_sec : /cone_pairs 타임아웃 [s] (기본 1.0)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # ── Launch 인자 선언 ──────────────────────────────────────────────────────
    args = [
        # 꼬깔 인식 파라미터
        DeclareLaunchArgument('camera_index',     default_value='0'),
        DeclareLaunchArgument('model_path',       default_value='/home/woohyuck/capstone_ws/src/cone_segment/model/best7.pt'),
        DeclareLaunchArgument('hfov_deg',         default_value='46.0'),
        DeclareLaunchArgument('conf_thresh',      default_value='0.85'),
        DeclareLaunchArgument('show_image',       default_value='true'),
        DeclareLaunchArgument('img_width',        default_value='640'),
        DeclareLaunchArgument('img_height',       default_value='480'),
        DeclareLaunchArgument('cone_real_height', default_value='0.23',
                              description='실제 꼬깔 높이 [m]'),
        DeclareLaunchArgument('max_disappeared',  default_value='45'),
        DeclareLaunchArgument('max_cost',         default_value='150.0'),
        # 속도 제어 파라미터
        DeclareLaunchArgument('u_min',            default_value='1.5',
                              description='최소 전진 속도 [m/s]'),
        DeclareLaunchArgument('u_max',            default_value='2.0',
                              description='최대 전진 속도 [m/s]'),
        DeclareLaunchArgument('speed_k',          default_value='1.0',
                              description='distance → speed 비례계수'),
        DeclareLaunchArgument('use_distance',     default_value='true',
                              description='거리 기반 가변 속도 사용 여부'),
        DeclareLaunchArgument('cmd_timeout_sec',  default_value='1.0',
                              description='/cone_pairs 타임아웃 [s]'),
    ]

    # ── ① cone_detector_node (conda torch 환경에서 실행) ─────────────────────
    torch_python = '/home/woohyuck/miniconda3/envs/torch/bin/python'
    node_script  = '/home/woohyuck/capstone_ws/src/cone_segment/cone_segment/cone_detector_node.py'

    cone_detector = ExecuteProcess(
        cmd=[
            torch_python, node_script,
            '--ros-args',
            '-r', '__node:=cone_detector_node',
            '-p', ['camera_index:=',     LaunchConfiguration('camera_index')],
            '-p', ['model_path:=',       LaunchConfiguration('model_path')],
            '-p', ['hfov_deg:=',         LaunchConfiguration('hfov_deg')],
            '-p', ['conf_thresh:=',      LaunchConfiguration('conf_thresh')],
            '-p', ['show_image:=',       LaunchConfiguration('show_image')],
            '-p', ['img_width:=',        LaunchConfiguration('img_width')],
            '-p', ['img_height:=',       LaunchConfiguration('img_height')],
            '-p', ['cone_real_height:=', LaunchConfiguration('cone_real_height')],
            '-p', ['max_disappeared:=',  LaunchConfiguration('max_disappeared')],
            '-p', ['max_cost:=',         LaunchConfiguration('max_cost')],
        ],
        output='screen',
    )

    # ── ② speed_controller_node (일반 ROS2 Node) ─────────────────────────────
    speed_controller = Node(
        package='cone_segment',
        executable='speed_controller_node',
        name='speed_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'u_min':           LaunchConfiguration('u_min'),
            'u_max':           LaunchConfiguration('u_max'),
            'speed_k':         LaunchConfiguration('speed_k'),
            'use_distance':    LaunchConfiguration('use_distance'),
            'cmd_timeout_sec': LaunchConfiguration('cmd_timeout_sec'),
        }],
    )

    return LaunchDescription(args + [cone_detector, speed_controller])