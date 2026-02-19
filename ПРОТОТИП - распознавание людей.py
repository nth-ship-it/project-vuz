import cv2
import numpy as np
import os
import urllib.request


def download_model_if_needed():
    """Скачивает модель если файлы отсутствуют"""
    prototxt_path = "detector/MobileNetSSD_deploy.prototxt"
    caffemodel_path = "detector/MobileNetSSD_deploy.caffemodel"

    # Создаем папку если её нет
    if not os.path.exists('detector'):
        os.makedirs('detector')

    # Скачиваем prototxt если отсутствует
    if not os.path.exists(prototxt_path):
        print("Скачивание prototxt файла...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
            prototxt_path
        )

    # Скачиваем caffemodel если отсутствует
    if not os.path.exists(caffemodel_path):
        print("Скачивание caffemodel файла...")
        urllib.request.urlretrieve(
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
            caffemodel_path
        )


def load_model():
    """Загружает модель MobileNetSSD"""
    prototxt = "detector/MobileNetSSD_deploy.prototxt"
    caffemodel = "detector/MobileNetSSD_deploy.caffemodel"

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    return net, classes


def detect_people(frame, net, classes):
    (h, w) = frame.shape[:2]

    # Создаем blob из изображения
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    people_boxes = []

    # Обрабатываем обнаружения
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:  # Минимальная уверенность
            class_id = int(detections[0, 0, i, 1])

            if classes[class_id] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Корректируем координаты чтобы не выйти за границы
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                people_boxes.append((startX, startY, endX, endY, confidence))

    return people_boxes


def main():
    # Скачиваем модель если нужно
    download_model_if_needed()

    # Загружаем модель
    print("Загрузка модели...")
    net, classes = load_model()
    print("Модель загружена!")

    # Подключаемся к камере
    cap = cv2.VideoCapture(0)  # 0 - встроенная, 1 - USB

    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере!")
        return

    print("Камера подключена. Нажмите 'q' для выхода.")
    print("Нажмите 'f' для переключения в полноэкранный режим.")

    # Создаем окно с возможностью изменения размера
    cv2.namedWindow("People Detection", cv2.WINDOW_NORMAL)

    # Устанавливаем начальный размер окна (почти на весь экран)
    cv2.resizeWindow("People Detection", 1200, 800)

    fullscreen = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: Не удалось получить кадр!")
            break

        # Получаем текущий размер окна
        window_size = cv2.getWindowImageRect("People Detection")
        if window_size[2] > 0 and window_size[3] > 0:  # Если окно существует
            # Изменяем размер кадра под размер окна
            frame_resized = cv2.resize(frame, (window_size[2], window_size[3]))
        else:
            # Если не удалось получить размер окна, используем стандартный размер
            frame_resized = cv2.resize(frame, (800, 600))

        # Обнаруживаем людей
        people_boxes = detect_people(frame_resized, net, classes)

        # Рисуем прямоугольники
        for (startX, startY, endX, endY, confidence) in people_boxes:
            cv2.rectangle(frame_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame_resized, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Добавляем счетчик
        cv2.putText(frame_resized, f"People: {len(people_boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Добавляем подсказку для полноэкранного режима
        cv2.putText(frame_resized, "Press 'f' for fullscreen, 'q' to quit",
                    (10, frame_resized.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Показываем результат
        cv2.imshow("People Detection", frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Переключаем полноэкранный режим
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("People Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("People Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow("People Detection", 1200, 800)

    cap.release()
    cv2.destroyAllWindows()


main()
