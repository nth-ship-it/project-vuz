import cv2
import numpy as np
import os
import urllib.request
import time
import traceback


def download_model_if_needed():
    """Скачивает модель если файлы отсутствуют с обработкой ошибок"""
    prototxt_path = "detector/MobileNetSSD_deploy.prototxt"
    caffemodel_path = "detector/MobileNetSSD_deploy.caffemodel"

    # Создаем папку если её нет
    if not os.path.exists('detector'):
        os.makedirs('detector')

    # Скачиваем prototxt если отсутствует
    if not os.path.exists(prototxt_path):
        print("Скачивание prototxt файла...")
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
                prototxt_path
            )
            print("prototxt файл успешно скачан")
        except Exception as e:
            print(f"Ошибка скачивания prototxt: {e}")
            return False

    # Скачиваем caffemodel если отсутствует
    if not os.path.exists(caffemodel_path):
        print("Скачивание caffemodel файла...")
        try:
            urllib.request.urlretrieve(
                "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
                caffemodel_path
            )
            print("caffemodel файл успешно скачан")
        except Exception as e:
            print(f"Ошибка скачивания caffemodel: {e}")
            return False

    return True


def load_model():
    """Загружает модель MobileNetSSD с обработкой ошибок"""
    prototxt = "detector/MobileNetSSD_deploy.prototxt"
    caffemodel = "detector/MobileNetSSD_deploy.caffemodel"

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        print("Файлы модели не найдены!")
        return None, None

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print("Модель успешно загружена!")
        return net, classes
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None


def detect_people(frame, net, classes, confidence_threshold=0.4):
    """Обнаружение людей с улучшенной логикой"""
    if frame is None or net is None:
        return []

    (h, w) = frame.shape[:2]

    # Создаем blob из изображения (точные параметры из исходного кода)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    people_boxes = []

    # Обрабатываем обнаружения
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            if classes[class_id] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Корректируем координаты чтобы не выйти за границы
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Фильтр слишком маленьких детекций (оптимизация)
                width = endX - startX
                height = endY - startY
                if width > 20 and height > 40:  # Минимальный размер человека
                    people_boxes.append((startX, startY, endX, endY, confidence))

    return people_boxes


def enhance_image_quality(frame):
    """Быстрое улучшение качества изображения"""
    if frame is None or frame.size == 0:
        return frame

    try:
        # Простое улучшение качества без изменения параметров камеры
        # Работаем в LAB пространстве для лучшего контроля яркости
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE для улучшения контраста только на канале яркости
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Легкое подавление шума
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        return enhanced

    except Exception:
        return frame


def main():
    # Скачиваем модель если нужно
    if not download_model_if_needed():
        print("Ошибка: не удалось скачать модель")
        return

    # Загружаем модель
    print("Загрузка модели...")
    net, classes = load_model()

    if net is None or classes is None:
        print("Ошибка: не удалось загрузить модель!")
        return

    print("Модель загружена!")

    # Подключаемся к камере (ТОЧНО КАК В ИСХОДНОМ КОДЕ)
    cap = cv2.VideoCapture(0)  # 0 - встроенная, 1 - USB
    # НИКАКИХ ДОПОЛНИТЕЛЬНЫХ НАСТРОЕК КАМЕРЫ!

    if not cap.isOpened():
        print("Ошибка: Не удалось подключиться к камере!")
        return

    print("Камера подключена. Нажмите 'q' для выхода.")
    print("Нажмите 'f' для переключения в полноэкранный режим.")
    print("Нажмите 'c' для включения/выключения улучшения качества")
    print("Нажмите '+' или '-' для изменения порога уверенности")

    # Создаем окно с возможностью изменения размера
    cv2.namedWindow("People Detection", cv2.WINDOW_NORMAL)

    # Устанавливаем начальный размер окна (почти на весь экран)
    cv2.resizeWindow("People Detection", 1200, 800)

    # Переменные состояния
    fullscreen = False
    enhance_enabled = False  # По умолчанию выключено для сохранения оригинального вида
    confidence_threshold = 0.4  # Как в исходном коде
    frame_count = 0
    fps = 0
    fps_update_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Ошибка: Не удалось получить кадр!")
                # Простая попытка продолжить работу
                time.sleep(0.1)
                continue

            # Улучшение качества если включено
            if enhance_enabled:
                frame = enhance_image_quality(frame)

            # Получаем текущий размер окна (как в исходном коде)
            window_size = cv2.getWindowImageRect("People Detection")
            if window_size[2] > 0 and window_size[3] > 0:  # Если окно существует
                # Изменяем размер кадра под размер окна
                frame_resized = cv2.resize(frame, (window_size[2], window_size[3]))
            else:
                # Если не удалось получить размер окна, используем стандартный размер
                frame_resized = cv2.resize(frame, (800, 600))

            # Обнаруживаем людей
            people_boxes = detect_people(frame_resized, net, classes, confidence_threshold)

            # Расчет FPS
            frame_count += 1
            current_time = time.time()
            if current_time - fps_update_time >= 1.0:
                fps = frame_count / (current_time - fps_update_time)
                frame_count = 0
                fps_update_time = current_time

            # Рисуем прямоугольники (стиль как в исходном коде, но с улучшениями)
            for (startX, startY, endX, endY, confidence) in people_boxes:
                # Цвет в зависимости от уверенности
                if confidence > 0.7:
                    color = (0, 255, 0)  # Зеленый
                elif confidence > 0.5:
                    color = (0, 200, 255)  # Оранжевый
                else:
                    color = (0, 100, 255)  # Красноватый

                cv2.rectangle(frame_resized, (startX, startY), (endX, endY), color, 2)

                # Текст с уверенностью (более информативный)
                label = f"Person: {confidence:.1%}"
                cv2.putText(frame_resized, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Добавляем счетчик (улучшенный)
            cv2.putText(frame_resized, f"People: {len(people_boxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Добавляем FPS
            cv2.putText(frame_resized, f"FPS: {fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Добавляем статус улучшения качества
            quality_status = "ON" if enhance_enabled else "OFF"
            cv2.putText(frame_resized, f"Quality: {quality_status}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Добавляем порог уверенности
            cv2.putText(frame_resized, f"Threshold: {confidence_threshold:.2f}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Добавляем подсказку для управления (расширенная)
            help_text = "Press: f-fullscreen, q-quit, c-quality, +/- - threshold"
            cv2.putText(frame_resized, help_text,
                        (10, frame_resized.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Показываем результат
            cv2.imshow("People Detection", frame_resized)

            # Обработка клавиш (расширенная функциональность)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Выход из программы...")
                break
            elif key == ord('f'):
                # Переключаем полноэкранный режим (точно как в исходном коде)
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty("People Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("People Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("People Detection", 1200, 800)
            elif key == ord('c'):
                # Переключаем улучшение качества
                enhance_enabled = not enhance_enabled
                status = "включено" if enhance_enabled else "выключено"
                print(f"Улучшение качества: {status}")
            elif key == ord('+'):
                # Увеличиваем порог уверенности
                confidence_threshold = min(0.9, confidence_threshold + 0.05)
                print(f"Порог уверенности увеличен до: {confidence_threshold:.2f}")
            elif key == ord('-'):
                # Уменьшаем порог уверенности
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"Порог уверенности уменьшен до: {confidence_threshold:.2f}")

            # Проверяем, не закрыто ли окно
            if cv2.getWindowProperty("People Detection", cv2.WND_PROP_VISIBLE) < 1:
                print("Окно закрыто пользователем")
                break

    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\nОшибка в основном цикле: {e}")
        traceback.print_exc()
    finally:
        # Освобождаем ресурсы
        print("\nОсвобождение ресурсов...")
        cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")

main()