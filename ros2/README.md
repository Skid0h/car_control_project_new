# Jetson AGX Xavier: ROS 2 Foxy

Нативный запуск: **Ubuntu 20.04 / ARM64, JetPack 5.1.3, L4T 35.5.0,
Python 3.8, CUDA 11.4, TensorRT 8.5.2**. Foxy завершил поддержку; он выбран
для сохранения этой ОС. Noetic можно оставить установленным, но не подключать
его окружение в терминале ROS 2. Обновлять ОС/CUDA не требуется.

Перенесён только Jetson. Пульт PC/server_send.py сохранён. ros/ содержит старые
ROS 1 симуляторы, Jetson Xavier/server.py — старый сервер. Для нового запуска
используйте команды ниже. Старый сервер одновременно запускать нельзя.

## Установка

Откройте чистый терминал без подключения Noetic. Уберите автоматический
source /opt/ros/noetic/setup.bash из .bashrc, если он там есть.

```bash
sudo apt-get update
sudo apt-get install -y curl gnupg locales software-properties-common
sudo locale-gen en_US en_US.UTF-8
export LANG=en_US.UTF-8
sudo add-apt-repository universe
gpg --keyserver keyserver.ubuntu.com --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA
gpg --export 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA \
  | sudo tee /usr/share/keyrings/ros2-snapshots-archive-keyring.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros2-snapshots-archive-keyring.gpg] http://snapshots.ros.org/foxy/final/ubuntu focal main" \
  | sudo tee /etc/apt/sources.list.d/ros2-snapshots.list
sudo apt-get update
sudo apt-get install -y ros-foxy-ros-base ros-foxy-launch-ros \
  python3-colcon-common-extensions python3-pytest python3-serial \
  python3-numpy python3-flask python3-pycuda
sudo usermod -aG dialout "$USER"
```

После изменения группы выйдите из сеанса и войдите снова.
Из-за EOL доступность пакетов Foxy может измениться. Не подменяйте их
пакетами Humble для другой Ubuntu.

Сохраните OpenCV/TensorRT из JetPack. Проверка:

```bash
/usr/bin/python3 -c "import cv2,numpy,tensorrt,pycuda.driver,pyzed.sl; print(cv2.__version__, numpy.__version__, tensorrt.__version__)"
```

Отсутствующий TensorRT Python устанавливайте пакетом python3-libnvinfer из
репозитория NVIDIA для вашего L4T. OpenCV — python3-opencv из репозиториев
системы. Не заменяйте их pip-пакетами opencv-python, tensorrt или новым numpy.
Старый TensorRT использует NumPy API, удалённые в 1.24; системный NumPy
Ubuntu 20.04 подходит. PyTorch/Ultralytics для запуска не нужны.

Сохраните работающий ZED SDK. Если его нет, выберите архив для JetPack 5/
L4T 35 и Python 3.8 на сайте Stereolabs. API устанавливается скриптом именно
этого SDK: /usr/bin/python3 /usr/local/zed/get_python_api.py.
SDK для JetPack 6/7 не является автоматической заменой.

## Модель и сборка

**Обученного cone_detector_v3.engine в репозитории нет.**
Раньше конфиг ссылался на файл на SSD Jetson. Укажите ваш файл через model_path.
Yolo/yolo11n.pt нельзя считать заменой обученной модели конусов.

Поддерживается raw YOLOv8/YOLO11 explicit batch: один RGB NCHW input,
один output [1,7,N] либо [1,N,7], без встроенного NMS/objectness YOLOv5.
Классы: **0=yellow, 1=orange, 2=blue** — сверьте с обучением.
Динамический профиль должен принимать 1x3x640x640; статический размер
берётся из engine. Сохранён preprocess: stretch; если обучение/экспорт
предполагает letterbox, задайте preprocess: letterbox в jetson.yaml.

Engine строится на совместимом Xavier/TensorRT 8.5.2. При наличии
соответствующего ONNX:
`/usr/src/tensorrt/bin/trtexec --onnx=cone_detector.onnx --saveEngine=cone_detector.engine --fp16`.
Для динамического ONNX нужны min/opt/maxShapes с реальным именем input.
Обучение/экспорт модели здесь не выполняются.

Из корня репозитория:

```bash
bash ros2/build.sh
source /opt/ros/foxy/setup.bash
source ros2/install/setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 run car_jetson preflight --model /absolute/path/cone_detector_v3.engine --camera
```

Preflight не открывает Serial. Проверяет платформу, импорты, engine,
синтетический инференс и с --camera получение кадра ZED. Этот тест
не доказывает точность распознавания.

## Первый запуск без приводов

```bash
ros2 launch car_jetson jetson.launch.py \
  model_path:=/absolute/path/cone_detector_v3.engine dry_run:=true
```

На PC запустите PC/server_send.py с правильным JETSON_IP.
UDP 5005, протокол и поля телеметрии сохранены.
dry_run по умолчанию true: Serial вообще не открывается.

```bash
ros2 topic echo /car/status
ros2 topic echo /car/command
ros2 topic hz /car/cones
```

/car/command — телеметрия сформированных команд, не вход управления.
throttle -1..1; руль положительный вправо; brake=1 означает нейтраль ESC,
не активный механический тормоз. Конусы: base_link, x вперёд, y влево, метры.

A — авто, S — стоп/ручной, стрелки — ручной перехват, R/C — запись/стоп,
Q — стоп и завершение узлов. F останавливает и предлагает перезапустить
launch: горячий рестарт USB/CUDA удалён. PC может менять скорость только
внутри лимитов конфигурации. После потери камеры/пульта требуется новое A.
Оранжевый финиш фиксируется до S → A.

Запись: аннотированное AVI/MJPG в ~/zed_recordings, без очереди старых кадров.
Медленный диск может вызвать защитную остановку. Веб-видео включается
web_enabled:=true: http://JETSON_IP:5000; по умолчанию отключено.

## Руль и Arduino

Исходный command_interval ограничивал только повторы; изменившиеся команды
отправлялись сразу. В старом Arduino-файле есть лишняя обратная косая черта,
мешающая компиляции, и аварийный центр 90 вместо 100.

Новый скетч: **firmware/car_controller/car_controller.ino**.
Соберите под фактическую плату и загрузите перед аппаратными испытаниями.
Пины: ESC 9, servo 10; 9600 baud. CI использует Uno для проверки AVR-синтаксиса,
это не подтверждение модели вашей платы.

Защита:
- все обычные Serial-команды максимум 20 Гц, без очереди;
- изменение целевого угла менее 2 командных градусов игнорируется относительно
  последней принятой цели; накопившаяся достаточная поправка принимается;
- скорость изменения 60 командных градусов/с, округление до целого;
- начальные пределы 70..130, центр 100, исходный масштаб ±50;
- heartbeat каждые 200 мс;
- отсутствие новой команды 300 мс → нейтраль двигателя;
- отсутствие свежих измерений 400 мс → выход из AUTO;
- потеря PC 500 мс → выход из AUTO;
- независимый watchdog Arduino 500 мс после корректного пакета;
- нейтраль обходит ограничение частоты; руль плавно возвращается к
  откалиброванному центру. Через 1,2 с прошивка отключает импульсы серво,
  снимая удерживающий момент. Машина может продолжать катиться по инерции.

В прошивке дополнительно ограничены частота и скорость руля, пакеты
проверяются целиком, servo.write вызывается только при изменении.
STOP передаётся как <1500,угол,S>; пульт PC по-прежнему использует прежний UDP.
Нулевые пакеты PC не отменяют STOP. Ручной поворот колёс при нулевом газе
сохранён после явной команды стрелкой; при потере связи прошивка сама
центрирует руль и отключает сервопривод.
Без новой прошивки отключение импульсов не работает, а при обрыве USB
сохраняется старый аварийный центр: Python не может исправить контроллер.

**100 — положение, не минимальная сила.** Отбрасывать всё меньше 100 нельзя.
Servo сама посылает импульсы примерно каждые 20 мс, даже без новых write().
Уменьшение Serial-трафика само по себе не отключает удерживающий момент.
После STOP обновлённая прошивка вызывает Servo.detach(), что снимает PWM;
питание серво остаётся включённым, поэтому это не замена механической
калибровке. На движущейся машине внезапное отключение руля может изменить
траекторию: сначала двигатель переводится в нейтраль, затем руль плавно
центрируется и только после задержки отключается.
Причину поломки по коду доказать нельзя: проверьте упоры, заклинивание тяги,
напряжение/пиковый ток питания и модель серво. Комментарии старого скетча
упоминают SC-15WP и HPI SF-10W, но это не проверка установленного оборудования.

**70..130 — стартовые, а не гарантированно безопасные пределы.**
Откалибруйте центр и края на вывешенных колёсах/с отсоединённой тягой без упора.
Задайте одинаковые значения в jetson.yaml и прошивке. Калибруйте ESC:
1500/1570/1420 — значения проекта, не универсальные безопасные скорости.
Параметры применяются при запуске, не через горячие ros2 param set.
Можно передать params:=/absolute/path/jetson.yaml и project_config:=/absolute/path/project.jsonc.

После проверки механики и прошивки:

```bash
ros2 launch car_jetson jetson.launch.py \
  model_path:=/absolute/path/cone_detector_v3.engine \
  serial_port:=/dev/serial/by-id/YOUR_DEVICE dry_run:=false
```

Укажите реальный by-id путь. Автопоиск допускает ровно одного контроллера.

## Проверка на месте и ограничения

1. С вывешенными колёсами проверьте направление, угол, плавность и S.
2. Закройте пульт во время команды: тяга должна прекратиться.
3. Проверьте AUTO без движения: старые/пустые измерения дают нейтраль;
   после потери камеры требуется новое A.
4. Отключите USB с вывешенными колёсами: аппаратный watchdog должен дать
   нейтраль в пределах 500 мс плюс время исполнения платы, затем плавно
   вернуть руль в центр и отключить импульсы сервопривода.
5. Проверьте цвета/боксы, R/C, tegrastats, частоту /car/cones, затем
   низкоскоростное движение на трассе.

Два процесса rclpy отделяют ZED/GPU от Serial/watchdog. Используются
монотонные часы и очереди глубиной 1; старые/повторные измерения отбрасываются.
При отсутствии конусов машина теперь останавливается. CUDA-контекст
сбалансирован, FP32/FP16 и число классов проверяются. Неиспользуемые настройки
удалены из нового JSONC.

Глубина по-прежнему 100/sqrt(площади бокса), не стереоглубина; нужна калибровка.
camera_right=0 сохраняет старую геометрию, положительное смещение вправо.
После аппаратной ошибки устраните причину и перезапустите launch.

Чистые тесты проверяют алгоритм, привод, PC-протокол и формы выходов детектора.
ROS-тест использует настоящий rclpy/UDP/сообщения после colcon; без ROS явно
пропускается. CI настроен на AMD64/ARM64 с Ubuntu 20.04/Python 3.8.
ARM64 CI не содержит GPU Jetson, ZED и приводов. Фактические результаты:
**VALIDATION.md**. Аппаратура, качество модели, углы и FPS требуют вашего испытания.

Источники: [ROS REP-2000](https://reps.openrobotics.org/rep-2000/),
[JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513),
[TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/index.html),
[Stereolabs](https://www.stereolabs.com/developers/release),
[Arduino Servo](https://github.com/arduino-libraries/Servo/blob/master/src/Servo.h).

Installation uses the official Foxy final snapshot, the same repository tested in CI.
Source: https://github.com/osrf/docker_images/blob/master/ros/foxy/ubuntu/focal/ros-core/Dockerfile
Do not enable a second conflicting ROS 2 apt source if one is already configured.
