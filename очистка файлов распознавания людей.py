import os
import shutil
import glob


def cleanup():
    print("Очистка файлов программы...")

    # Удаляем папку detector
    if os.path.exists("detector"):
        print(f"Удаление папки: detector")
        shutil.rmtree("detector")

    # Удаляем все файлы detection_*.jpg
    detection_files = glob.glob("detection_*.jpg")
    for file in detection_files:
        print(f"Удаление файла: {file}")
        os.remove(file)

    # Удаляем возможные другие файлы
    other_files_to_remove = [
        "people_detection.log",
        "error.log",
        "*.png",
        "*.tmp"
    ]

    for pattern in other_files_to_remove:
        for file in glob.glob(pattern):
            try:
                print(f"Удаление файла: {file}")
                os.remove(file)
            except:
                pass

    print("Очистка завершена!")


if __name__ == "__main__":
    cleanup()