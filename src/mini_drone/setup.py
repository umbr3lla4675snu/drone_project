from setuptools import find_packages, setup

package_name = 'mini_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/crazyflie_bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rubis',
    maintainer_email='rubis@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_deck_camera = mini_drone.ai_deck_camera_node:main',
            'cf_bridge = mini_drone.cf_bridge_node:main',
            'cf_keyboard_control_node = mini_drone.cf_keyboard_control_node:main',
            'person_counter = mini_drone.person_counter:main',
        ],
    },
)
