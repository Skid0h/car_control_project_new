# Jetson: ROS 2 Foxy / JetPack 5.1.3

Нативный ROS 2 контур для Jetson AGX Xavier: **[инструкция запуска и защиты приводов](ros2/README.md)**. PC-пульт сохранён. Начните с dry_run; требуется ваша обученная TensorRT-модель конусов.

Ниже — прежнее описание проекта и ROS 1 симуляторов.

# car_control_project_new_Danis — ветка ros

Существующий запуск машинки: `Jetson Xavier/server.py` (ZED + TensorRT + Arduino).

Добавлен ROS1-пакет с автономным лёгким симулятором, адаптером Formula Student
Driverless Simulator и входом камеры для существующего TensorRT-детектора.

**[Установка, запуск на Jetson/Linux, FSDS и тесты → ros/README.md](ros/README.md)**

После установки зависимостей:

```bash
bash ros/build.sh
source ros_ws/devel/setup.bash
roslaunch car_control_ros sim.launch
```

Это тест алгоритма на кинематической модели, а не доказательство готовности к
автономному движению реальной машины. ROS-узлы этой ветки не открывают Arduino.
