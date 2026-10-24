# Симуляция машинки через ROS

Основа: только `Main` вашего форка, коммит `bf04463934ccc88df5d29e100f76d0bbd4124a0f`.
Пакет добавлен в ветке `ros`; исходный `server.py` и аппаратные модули сохранены.

## Что реализовано

| Режим | Откуда конусы | Что проверяет | Где запускать |
|---|---|---|---|
| `sim.launch` | Геометрическая модель кольцевой трассы | Контроллер, ROS-сообщения, watchdog, остановка | Jetson или Linux, без GPU/модели/ZED |
| `fsds.launch perception:=ground_truth` | Положения конусов и одометрия FSDS | Контроллер на физике и трассе FSDS | Алгоритм на Jetson, FSDS + bridge на Linux-машине с GPU |
| `fsds.launch perception:=camera` | Изображение FSDS → ваш TensorRT-детектор | Детекция, оценка дальности, контроллер | Детектор на Jetson с вашей `.engine` |

Контроллер переносит из `VisionLoop` интерполяцию синих/жёлтых границ, выбор
lookahead на сетке, EMA и PID угла на цель. Это **порт алгоритма**, не запуск
исходного `server.py` в эмуляторе: исходный аппаратный цикл пока не использует
новый `core.py`. Изменения алгоритма далее нужно переносить осознанно; записи
веб-камеры, UDP-пульт и веб-интерфейс не входят в ROS-пакет.

Отличия ROS-порта: без пригодных конусов — торможение; устаревшие кадры не
обновляют управление; производная PID ограничена по dt; финиш фиксируется до
нового включения. Данные для watchdog проверяются по исходному ROS timestamp
и локальным монотонным часам. FSDS-адаптер также останавливается без GO-сигнала
и при пропадании команд. Если завершён **сам адаптер** через SIGKILL, отправить
торможение уже некому: используйте аварийную остановку на стороне FSDS.

## Версия Linux и ROS

Базовая конфигурация: **Jetson Xavier/NX, JetPack 5.x, Ubuntu 20.04,
ROS1 Noetic, системный Python 3.8**. JetPack 5.1.4 содержит Ubuntu 20.04 и
TensorRT 8.5.2 ([NVIDIA](https://developer.nvidia.com/embedded/jetpack-sdk-514)).
Это соответствует binding API существующего `Cone_detector.py` и ROS1-мосту
присланного FSDS. Noetic завершил поддержку 31 мая 2025 года
([Open Robotics](https://www.ros.org/blog/noetic-eol/)); выбор сделан для
совместимости с этим проектом, не как рекомендация для нового долгосрочного стека.

На JetPack 4 / Ubuntu 18.04 или JetPack 6 / Ubuntu 22.04 эти команды установки
не подходят. Узнайте версию через `cat /etc/os-release` и
`dpkg-query -W nvidia-jetpack`. ROS2 здесь не реализован; ROS1-сообщения нельзя
подключить напрямую к `fsds_ros2_bridge`. Нельзя обещать запуск на произвольном
Jetson без сведений о его JetPack. Перепрошивать Jetson для проверки не требуется.

## Быстрый запуск лёгкого симулятора

Установите ROS Noetic по [официальной инструкции](https://wiki.ros.org/noetic/Installation/Ubuntu).
Далее в терминале Ubuntu 20.04 с настроенным ROS apt-репозиторием:

```bash
sudo apt-get update
sudo apt-get install -y ros-noetic-ros-base ros-noetic-visualization-msgs \
  ros-noetic-cv-bridge ros-noetic-rostest python3-catkin-pkg python3-rospkg
git clone --single-branch --branch ros https://github.com/salimadze2005-beep/car_control_project_new_Danis.git
cd car_control_project_new_Danis
bash ros/build.sh
source ros_ws/devel/setup.bash
roslaunch car_control_ros sim.launch
```

Симулятор стартует в AUTO, не требует `fs_msgs`, CUDA, камеры или Arduino.
Кольцо имеет радиус 8 м и ширину 1.5 м; велосипедная модель: база 0.32 м,
максимальный угол колёс 0.44 рад, максимальная скорость 2 м/с. Это примерные
параметры, **не измеренная модель вашей машинки**. Нет проскальзывания,
модели подвески, задержек сервопривода и столкновений. Конусы идеальные,
ограничены дальностью и полем зрения; этот режим не проверяет YOLO.

Для вида сверху установите `sudo apt-get install ros-noetic-rviz`, затем:

```bash
roslaunch car_control_ros sim.launch rviz:=true
```

В другом терминале после `source ros_ws/devel/setup.bash`:

```bash
rostopic echo /car/status
rostopic hz /car/cones
rostopic echo /car/command
rosservice call /car/enable "data: false"
rosservice call /car/enable "data: true"
```

`/car/enable` — новый ROS-интерфейс включения. Исходный UDP-пульт к этим
узлам не подключён. Одновременно запускайте только один perception-источник
и один контроллер: имена `/car/*` общие для всех режимов.

## Подключение готового FSDS

Рекомендуемая схема: FSDS/Unreal и `fsds_ros_bridge` на отдельном **Linux x86_64
компьютере с GPU**, наши узлы — на Jetson. Это позволяет проверять алгоритмы
на целевом железе. ARM-сборка Unreal/FSDS на Jetson здесь не подготовлена.
Архитектура FSDS описана [здесь](https://fs-driverless.github.io/Formula-Student-Driverless-Simulator/v2.2.0/system-overview/).

1. На машине симулятора установите готовый FSDS и ROS1 bridge по
   [инструкции upstream](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator/blob/master/docs/getting-started-with-ros.md).
   Версия bridge должна совпадать с бинарником симулятора: для 2.2.0 берите
   соответствующую версию исходников, а не произвольный свежий master.
   Репозиторий bridge ожидается по пути `~/Formula-Student-Driverless-Simulator`.
   Bridge и Unreal должны использовать одинаковый `settings.json`.
   FSDS остаётся внешней зависимостью; его GPL-код/ассеты в этот форк не копировались.
2. На Jetson нужен только пакет сообщений `fs_msgs` из **того же checkout FSDS**,
   который используется на машине симулятора. Перенесите каталог
   `ros/src/fs_msgs` из установленного FSDS в `ros_ws/src/fs_msgs` вашего проекта.
   Если он пуст, на машине FSDS выполните `git submodule update --init ros/src/fs_msgs`.
   Пересоберите `bash ros/build.sh`. Для лёгкого симулятора этот шаг не нужен.
   Другие репозитории команды для разработки интеграции не использовались.
3. Синхронизируйте часы обеих машин (например, chrony), обеспечьте двустороннюю
   достижимость адресов. ROS1 использует 11311 для master и динамические TCP-порты
   для узлов. Откройте их внутри своей тестовой сети. Адреса ниже — примеры.

На Jetson, в **каждом** терминале ROS:

```bash
export ROS_MASTER_URI=http://192.168.1.20:11311
export ROS_IP=192.168.1.20
source ros_ws/devel/setup.bash
roscore
```

На Linux-машине FSDS (`192.168.1.10`), после запуска самого симулятора:

```bash
export ROS_MASTER_URI=http://192.168.1.20:11311
export ROS_IP=192.168.1.10
source ~/Formula-Student-Driverless-Simulator/ros/devel/setup.bash
roslaunch fsds_ros_bridge fsds_ros_bridge.launch host:=localhost competition_mode:=false manual_mode:=false
```

На Jetson, во втором настроенном терминале:

```bash
roslaunch car_control_ros fsds.launch perception:=ground_truth
```

В третьем:

```bash
rostopic hz /fsds/testing_only/odom
rostopic echo -n 1 /fsds/testing_only/track
rostopic hz /fsds/signal/go
rostopic hz /car/cones
rosservice call /car/enable "data: true"
```

Только после GO и свежих конусов будут выдаваться ненулевые команды. Для
остановки вызовите `/car/enable` с `false`. Финиш по оранжевому конусу в FSDS
выключен: там оранжевые конусы обозначают также стартовую область. Автоматический
подсчёт кругов и отправка `signal/finished` в этой версии не реализованы.
После `/fsds/reset` выключите и снова включите `/car/enable`, чтобы сбросить PID.

`testing_only` недоступны в competition_mode: это режим проверки управления
по известным конусам, а не оценка автономного восприятия.

Настройки для полноразмерного FSDS выделены в `config/fsds.yaml`: lookahead 5 м,
ширина 3 м, малый газ 0.08. Это начальные значения для настройки на вашей трассе;
не гарантируется прохождение любой трассы с ними. Газ — доля команды, не скорость
в м/с. Регулятор скорости по `/fsds/gss` пока не реализован.

## Проверка детекции по изображению FSDS

В репозитории **нет** `cone_detector_v3.engine`: исходная конфигурация указывает
на файл `/mnt/ArdorSSD/...` на Jetson. `Yolo/yolo11n.pt` не является подтверждённой
обученной моделью конусов и не подставляется вместо него. Нужен ваш engine,
совместимый с TensorRT/CUDA на этом Jetson, с 3 классами из `config.jsonc`.

Установите PyCUDA, OpenCV и TensorRT для вашего JetPack; `cv_bridge` установлен
выше. Проверьте в системном Python: `python3 -c 'import tensorrt, pycuda.driver, cv2'`.
Сначала остановите предыдущий `fsds.launch`, затем:

```bash
roslaunch car_control_ros fsds.launch perception:=camera \
  model_path:=/absolute/path/to/cone_detector_v3.engine
```

Проверьте `/fsds/camera/cam1` и `/car/cones`, затем включите `/car/enable`.
Этот режим не подписывается на ground truth: публикация конусов идёт только
через реальную детекцию изображения. Изображения RGB/BGRA переводятся в BGR
через `cv_bridge`. Очередь содержит только последний кадр. Ошибка детектора
или задержка более 0.4 с приводит к остановке, а не к повтору старой команды.

Оценка дальности сохраняет подход проекта: `constant / sqrt(bbox_area)`.
В `config/fsds_camera.yaml` число 300 — **некалиброванный стартовый коэффициент**.
По конусу на известном расстоянии измерьте `constant = distance * sqrt(area)`.
Повторите на нескольких расстояниях. Исходное число 100 для ZED автоматически
не переносится на полноразмерные конусы и другое разрешение FSDS.
`horizontal_fov_deg`, `camera_forward`, `camera_left` должны соответствовать
вашему `settings.json`; стартовые значения взяты из cam1 upstream master.
Поддерживается камера без поворота относительно автомобиля. Если CameraInfo
есть и его размер совпадает с кадром, берутся его fx/cx; иначе используется
заданный FOV. ZED SDK этот режим не требует.

## Интерфейсы и знаки

| Топик | Тип | Смысл |
|---|---|---|
| `/car/cones` | `car_control_ros/ConeArray` | `base_link`, метры, x вперёд / y влево |
| `/car/command` | `car_control_ros/Drive` | throttle/brake 0..1, steering -1..1, плюс **вправо** |
| `/car/status` | `std_msgs/String` | JSON enabled/sensor_fresh/finished/braking |
| `/car/sim/odom` | `nav_msgs/Odometry` | Положение лёгкой модели в map |
| `/car/sim/markers` | `visualization_msgs/MarkerArray` | Трасса и машинка для RViz |
| `/fsds/control_command` | `fs_msgs/ControlCommand` | Выход адаптера в FSDS |

Синий считается левой границей, жёлтый — правой, как в вычислениях текущего
`server.py` (некоторые комментарии `config.jsonc` говорят обратное). Внутри core
координаты камеры: right = -ROS_y, forward = ROS_x. Положительный руль совпадает
с FSDS: вправо. Нельзя подавать Drive как стандартный `/cmd_vel` или Ackermann:
здесь нормированные значения, не м/с и не радианы.

## Проверки

Без ROS/CUDA:

```bash
python3 -m unittest discover -s ros/car_control_ros/test -p 'test_*.py' -v
```

9 тестов: прямой/левый/правый руль, одна граница, отсутствие и некорректные конусы,
оранжевый финиш, часы/команды, преобразование координат, исходный Config,
замкнутое движение более двух кругов с отклонением менее 0.5 м.

С ROS, после сборки:

```bash
cd ros_ws
source devel/setup.bash
catkin_make run_tests
catkin_test_results
```

ROS-тесты проверяют реальную публикацию сообщений, движение, enable/disable,
остановку после исчезновения данных, старые timestamps и пустые детекции.
Workflow `.github/workflows/ros.yml` запускает эти проверки в Ubuntu 20.04 / Noetic.
Это не GPU-тест FSDS и не тест аппаратного Jetson. Проверка FSDS/камеры требует
вашего симулятора и engine; полного проезда FSDS и запуска на Jetson в этой
среде не было. Для приёмки последовательно проверьте лёгкий режим, FSDS ground
truth, FSDS camera, потерю камеры/bridge и остановку, затем калибровку модели
под размеры и управление реальной машинки.
