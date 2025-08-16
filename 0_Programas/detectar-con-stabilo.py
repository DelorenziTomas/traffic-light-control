import cv2
from ultralytics import YOLO
import sys
import math
import os
from stabilo import Stabilizer
from collections import deque

# Carga tu modelo personalizado
# Asegúrate de que la ruta a tu modelo 'best.pt' sea correcta
model_path = '/home/tdelorenzi/1_yolo/runs/detect/train8/weights/best.pt'
try:
    model = YOLO(model_path)
    print(f"Modelo cargado exitosamente: {model_path}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)

# --- CONFIGURACIÓN PARA EL CÁLCULO DE VELOCIDAD ---
PIXELES_POR_METRO = 250.8 # ¡Ajusta este valor según tu calibración!
posiciones_previas = {}
timestamps_previos = {}

# --- CONFIGURACIÓN PARA LA ESTELA ---
puntos_estela = {}
LARGO_MAX_ESTELA = 15 # Longitud de la estela en número de puntos (ajústalo a tu gusto)
RADIO_MAXIMO = 9       # Radio del círculo más nuevo y grande
RADIO_MINIMO = 1       # Radio del círculo más viejo y pequeño

# --- INICIALIZACIÓN DEL ESTABILIZADOR ---
stabilizer = Stabilizer()
is_ref_frame_set = False

# --- CLASES A DETECTAR ---
# Define las clases de tu modelo que quieres detectar y su color asociado
classes_to_find = {
    "people": {"name": "Persona", "color": (0, 255, 255)},
    "bicycle": {"name": "Bicicleta", "color": (255, 0, 255)},
    "car": {"name": "Auto", "color": (255, 0, 0)},
    "van": {"name": "Furgoneta", "color": (255, 255, 0)},
    "bus": {"name": "Autobus", "color": (0, 0, 255)},
    "motor": {"name": "Moto", "color": (0, 255, 0)}
}

# Mapeo automático de nombres de clases a IDs del modelo
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

# --- CONFIGURACIÓN DE VIDEO DE ENTRADA Y SALIDA ---
# Asegúrate de que la ruta a tu video sea correcta
video_path = '/home/tdelorenzi/1_yolo/2_CITTR/Videos/0.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta: {video_path}")
    sys.exit(1)

# Obtener información del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video cargado: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames a {fps} FPS, Dimensiones: {frame_width}x{frame_height}")

# Configuración del video de salida
output_dir = '/home/tdelorenzi/1_yolo/2_CITTR/3_Resultados'
os.makedirs(output_dir, exist_ok=True)
video_basename = os.path.basename(video_path)
video_filename, _ = os.path.splitext(video_basename)
output_path = os.path.join(output_dir, f"{video_filename}_resultado_final.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"El video procesado se guardará en: {output_path}")

# --- VENTANA Y CONTROLES ---
window_name = "Detección con Estela (Estabilizado) 🚗💨"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_count = 0
print("\nProcesando video... Controles disponibles:")
print("- 'q' o ESC: Salir y GUARDAR el video")
print("- Barra espaciadora: Pausar/reanudar")
print("- 'f': Alternar pantalla completa/ventana\n")

# ==============================================================================
# BUCLE PRINCIPAL DE PROCESAMIENTO
# ==============================================================================
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Fin del video o error al leer fotograma.")
            break

        frame_count += 1
        
        # --- BLOQUE DE ESTABILIZACIÓN DE VIDEO ---
        if not is_ref_frame_set:
            stabilizer.set_ref_frame(frame)
            is_ref_frame_set = True
        
        stabilizer.stabilize(frame)
        frame_estabilizado = stabilizer.warp_cur_frame()

        # A partir de aquí, todo se hace sobre el fotograma estabilizado
        results = model.track(frame_estabilizado, classes=classes_to_detect, persist=True, verbose=False)
        annotated_frame = frame_estabilizado.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls in label_map:
                    x1, y1, x2, y2 = map(int, box)
                    centro_x, centro_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    details = label_map[cls]
                    color = details["color"]
                    label_name = details["name"]

                    # --- LÓGICA DE LA ESTELA DE CÍRCULOS ---
                    if track_id not in puntos_estela:
                        puntos_estela[track_id] = deque(maxlen=LARGO_MAX_ESTELA)
                    puntos_estela[track_id].append((centro_x, centro_y))

                    num_puntos = len(puntos_estela[track_id])
                    for i, punto in enumerate(puntos_estela[track_id]):
                        if num_puntos > 1:
                            progreso = i / (num_puntos - 1)
                            radio_actual = int(RADIO_MINIMO + (RADIO_MAXIMO - RADIO_MINIMO) * progreso)
                        else:
                            radio_actual = RADIO_MAXIMO
                        
                        cv2.circle(annotated_frame, punto, radio_actual, color, -1)
                    
                    # --- CÁLCULO DE VELOCIDAD ---
                    velocidad_kmh = 0
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
                    
                    # --- DIBUJO DE RECTÁNGULO Y ETIQUETA ---
                    label_text = f"ID {track_id} {label_name} {velocidad_kmh:.1f} km/h"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x, text_y = x1, y1 - 10
                    cv2.rectangle(annotated_frame, (text_x, text_y - text_height), (text_x + text_width, text_y + 5), color, -1)
                    cv2.putText(annotated_frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- MOSTRAR Y GUARDAR FOTOGRAMA ---
        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)
        
        # --- MANEJO DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Saliendo y guardando el video...")
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('f'):
            current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if current_state == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Modo ventana activado")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Modo fullscreen activado")

except Exception as e:
    print(f"Ocurrió un error inesperado durante el procesamiento: {e}")

finally:
    # --- LIBERAR RECURSOS ---
    if cap.isOpened():
        cap.release()
    if out.isOpened():
        out.release()
    cv2.destroyAllWindows()
    # Espera 1ms para asegurar que las ventanas se cierren correctamente en todos los S.O.
    cv2.waitKey(1) 

print(f"\nProcesamiento finalizado. Video guardado en {output_path}")