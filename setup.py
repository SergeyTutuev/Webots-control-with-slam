#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'car_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        
        # Resources (images, templates, etc.)
        (os.path.join('share', package_name, 'resources'), 
         glob('resources/*.png') + glob('resources/*.jpg') + glob('resources/*.jpeg')),
        
        # Launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py') + glob('launch/*.py')),
        
        # Augmentation script (moved into package `scripts/`)
        (os.path.join('share', package_name, 'scripts'), ['scripts/create_augmented.py']),
        
        # ВАЖНО: копируем содержимое папки city_traffic_net
        (os.path.join('share', package_name, 'world', 'city_traffic_net'), glob('world/city_traffic_net/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # ВАЖНО: копируем файлы из корня world (например, city_traffic.wbt), но ИГНОРИРУЕМ папки
        (os.path.join('share', package_name, 'world'), 
         [f for f in glob('world/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='Leonardo Chen',
    maintainer_email='leochenjin@gmail.com',
    description='Car package with sign detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webots_bridge = car_pkg.webots_bridge:main',
            'point_navigator = car_pkg.point_navigator:main',
            'room_mapper = car_pkg.room_mapper:main'
        ],
    },
)
