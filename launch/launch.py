#!/usr/bin/env python3

"""ROS2 launch description with dual-mode selection."""

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import sys


def _prompt_mode() -> str:
    """Спрашивает режим у пользователя в stdin. Возвращает '1' или '2'."""
    print("")
    print("=" * 55)
    print("  Выберите режим работы робота")
    print("    1 — обход помещения и построение карты (Nav2)")
    print("    2 — движение к заданной точке (Bug algorithm)")
    print("=" * 55)
    while True:
        try:
            choice = input("Введите 1 или 2: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nРежим не выбран, использую по умолчанию (1).")
            return '1'
        if choice in ('1', '2'):
            return choice
        print("Некорректный ввод, попробуйте снова.")


def _launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)

    if mode not in ('1', '2'):
        mode = _prompt_mode()

    nodes_to_launch = []

    if mode == '1':
        selected_exec = 'room_mapper'
        mode_name = 'обход помещения + построение карты (Bug Algo)'
    else:
        selected_exec = 'point_navigator'
        mode_name = 'движение к заданной точке (Bug algorithm)'

    # В ОБОИХ РЕЖИМАХ ТЕПЕРЬ ИСПОЛЬЗУЕТСЯ ПРЯМОЕ УПРАВЛЕНИЕ
    use_nav2 = False 

    print(f"\n>>> Запуск режима {mode}: {mode_name}\n")

    bridge_node = Node(
        package='car_pkg',
        executable='webots_bridge',
        name='webots_bridge',
        parameters=[{'use_nav2': use_nav2}, {'use_sim_time': False}],
        additional_env={'WEBOTS_CONTROLLER_URL': 'tcp://127.0.0.1:1234'},
    )
    nodes_to_launch.append(bridge_node)

    nav_node = Node(
        package='car_pkg',
        executable=selected_exec,
        name=selected_exec,
        parameters=[{'use_sim_time': True}]
    )
    nodes_to_launch.append(nav_node)

    return nodes_to_launch

def generate_launch_description():
    script_path = os.path.join(
        get_package_share_directory('car_pkg'),
        'scripts', 'create_augmented.py',
    )
    pre_generate = ExecuteProcess(
        cmd=[sys.executable, script_path], output='screen'
    )

    declare_webots_arg = DeclareLaunchArgument(
        'start_webots', default_value='true',
        description='Запускать симулятор Webots',
    )
    declare_mode_arg = DeclareLaunchArgument(
        'mode', default_value='',
        description="Режим: 1=обход, 2=точка. Пусто — интерактивный запрос.",
    )

    world_file = os.path.join(
        get_package_share_directory('car_pkg'), 'world', 'break_room.wbt'
    )
    webots_proc = ExecuteProcess(
        cmd=['webots', world_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_webots')),
    )

    return LaunchDescription([
        declare_webots_arg,
        declare_mode_arg,
        pre_generate,
        webots_proc,
        OpaqueFunction(function=_launch_setup),
    ])