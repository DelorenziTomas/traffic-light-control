from ultralytics import YOLO

model = YOLO('yolo11x.pt')

# Entrenamiento con carpeta personalizada
results = model.train(
    data='/home/tdelorenzi/1_yolo/2_CITTR/1_Dataset/4/data.yaml',
    epochs=300,
    imgsz=640,
    batch=4,
    project='/home/tdelorenzi/1_yolo/2_CITTR/2_Modelos/roboflow',  # Ruta ABSOLUTA
    name='train',  # Nombre de la subcarpeta (obligatorio)
    exist_ok=True  # Evita errores si la carpeta existe
)