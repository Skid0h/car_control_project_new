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
