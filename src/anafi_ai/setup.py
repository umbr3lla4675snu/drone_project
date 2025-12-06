from setuptools import find_packages, setup

package_name = 'anafi_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rubis',
    maintainer_email='dlghdydwkd79@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'anafi_keyboard_control = anafi_ai.anafi_keyboard_control:main',
            'person_follower = anafi_ai.person_follower:main',
        ],
    },
)
