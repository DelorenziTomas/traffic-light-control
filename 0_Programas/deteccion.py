import cv2
from ultralytics import YOLO
import sys
import math
import os

# Carga tu modelo personalizado
model_path = '/home/tdelorenzi/1_yolo/runs/detect/train8/weights/best.pt'
try:
    model = YOLO(model_path)
    print(f"Modelo cargado exitosamente: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# --- CONFIGURACIÓN PARA EL CÁLCULO DE VELOCIDAD ---
PIXELES_POR_METRO = 152.8 # ¡¡¡AJUSTA ESTE VALOR!!!
posiciones_previas = {}
timestamps_previos = {}

# Búsqueda dinámica para MÚLTIPLES clases
classes_to_find = {
    "car": {"name": "Auto", "color": (255, 0, 0)},
    "motor": {"name": "Moto", "color": (0, 255, 0)}
}
try:
    names_inv = {v: k for k, v in model.names.items()}
except AttributeError:
    print("Error: No se pudo acceder a los nombres de las clases del modelo.")
    sys.exit(1)
label_map = {}
for class_name, details in classes_to_find.items():
    if class_name in names_inv:
        class_id = names_inv[class_name]
        label_map[class_id] = details
        print(f"Clase '{class_name}' encontrada con ID: {class_id}. Se traducirá a '{details['name']}'.")
    else:
        print(f"Advertencia: La clase '{class_name}' no se encontró en el modelo.")
if not label_map:
    print("Error: Ninguna de las clases deseadas se encontró en el modelo. Abortando.")
    sys.exit(1)
classes_to_detect = list(label_map.keys())

# Ruta de tu video
video_path = '/home/tdelorenzi/1_yolo/2_CITTR/Videos/0.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta: {video_path}")
    sys.exit(1)

# Obtener información del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video cargado: {total_frames} frames a {fps} FPS, Dimensiones: {frame_width}x{frame_height}")


# Configuración del video de salida
output_dir = '/home/tdelorenzi/1_yolo/2_CITTR/3_Resultados'
os.makedirs(output_dir, exist_ok=True) 

video_basename = os.path.basename(video_path)
video_filename, video_ext = os.path.splitext(video_basename)
output_path = os.path.join(output_dir, f"{video_filename}_resultado.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"El video procesado se guardará en: {output_path}")


# Ventana y controles
window_name = "Detección y Velocidad de Vehículos 🚗💨"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_count = 0
print("Procesando video... Controles disponibles:")
print("- 'q' o ESC: Salir y GUARDAR el video")
print("- Barra espaciadora: Pausar/reanudar")
print("- 'f': Alternar fullscreen/ventana")

# Bucle principal del video
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Fin del video o error al leer fotograma. Guardando el video procesado...")
            break

        frame_count += 1
        
        results = model.track(frame, classes=classes_to_detect, persist=True, verbose=False)
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls in label_map:
                    # ... (toda la lógica de cálculo de velocidad y dibujo no cambia) ...
                    x1, y1, x2, y2 = map(int, box)
                    velocidad_kmh = 0
                    centro_x, centro_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    if track_id in posiciones_previas:
                        pos_prev = posiciones_previas[track_id]
                        tiempo_prev = timestamps_previos[track_id]
                        distancia_pixeles = math.sqrt((centro_x - pos_prev[0])**2 + (centro_y - pos_prev[1])**2)
                        distancia_metros = distancia_pixeles / PIXELES_POR_METRO
                        tiempo_segundos = (frame_count - tiempo_prev) / fps
                        if tiempo_segundos > 0:
                            velocidad_kmh = (distancia_metros / tiempo_segundos) * 3.6

                    posiciones_previas[track_id] = (centro_x, centro_y)
                    timestamps_previos[track_id] = frame_count
                    
                    details = label_map[cls]
                    color = details["color"]
                    label_name = details["name"]
                    label_text = f"ID {track_id} {label_name} {velocidad_kmh:.1f} km/h"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = x1 + 5
                    text_y = y1 + text_height + 5
                    cv2.rectangle(annotated_frame, (x1, y1), (x1 + text_width + 10, y1 + text_height + 10), color, -1)
                    cv2.putText(annotated_frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Saliendo y guardando el video...")
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        # --- CÓDIGO CORREGIDO AQUÍ ---
        elif key == ord('f'):
            current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if current_state == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Modo ventana activado")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Modo fullscreen activado")
        # --- FIN DE LA CORRECCIÓN ---

except Exception as e:
    print(f"Error durante el procesamiento: {e}")

finally:
    if cap.isOpened():
        cap.release()
    if out.isOpened():
        out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

print(f"Procesamiento finalizado. Video guardado en {output_path}")