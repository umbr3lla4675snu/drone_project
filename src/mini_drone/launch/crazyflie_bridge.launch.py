from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mini_drone',
            executable='cf_bridge',
            name='cf_bridge',
            output='screen',
            parameters=[{
                'uri': 'radio://0/80/2M/E7E7E7E706',
                'period_ms': 100,
                'publish_rate_hz': 20.0,
                'use_state_estimate': True
            }],
        ),
        Node(
            package='mini_drone',
            executable='ai_deck_camera',
            name='ai_deck_camera',
            output='screen',
            parameters=[{
                # 예시: MJPEG 스트림 URL 또는 '0' (내장카메라)
                'camera_source': '0',
                'fps': 20.0,
            }],
        ),
    ])

