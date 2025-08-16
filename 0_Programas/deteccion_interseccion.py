import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, List

# Configuración centralizada
CONFIGURACION: Dict[str, Any] = {
    "ruta_video": Path('/home/tdelorenzi/1_yolo/2_CITTR/Videos/0.mp4'),
    "ruta_modelo": '/home/tdelorenzi/1_yolo/2_CITTR/2_Modelos/train/weights/best.pt',
    "umbral_confianza": 0.4,
    "clases_vehiculos": [0, 1, 2, 3, 5, 7],  # COCO classes: car, motorcycle, bus, truck
    "zonas_interes": {
        "Norte": np.array([[1503, 2], [2008, 2], [2008, 361], [1503, 361]], np.int32),
        "Sur": np.array([[1881, 1848], [2398, 1848], [2398, 2155], [1888, 2157]], np.int32),
        "Este": np.array([[2659, 675], [3830, 686], [3837, 1242], [2643, 1219]], np.int32),
        "Oeste": np.array([[2, 1014], [1270, 1039], [1243, 1560], [2, 1574]], np.int32)
    },
    "colores_zonas": {
        "Norte": (0, 255, 0),    # Verde
        "Sur": (0, 0, 255),      # Rojo
        "Este": (255, 255, 0),   # Cian
        "Oeste": (0, 255, 255)   # Amarillo
    },
    # Nuevas configuraciones de estilo
    "estilo_bounding_box": {
        "color": (255, 0, 255),  # Magenta (B, G, R)
        "grosor": 5
    },
    "estilo_zonas": {
        "grosor": 4
    },
    "estilo_texto_confianza": {
        "color": (255, 255, 0),  # Cian (B, G, R)
        "grosor": 2
    }
}

class ContadorDensidadVehiculos:
    def __init__(self, configuracion: Dict[str, Any]):
        self.configuracion = configuracion
        
        # Cargar modelo YOLOv11
        self.modelo = YOLO(configuracion["ruta_modelo"])
        
        # Configuración de video
        self.captura = cv2.VideoCapture(str(configuracion["ruta_video"]))
        if not self.captura.isOpened():
            raise IOError(f"Error al abrir el video: {configuracion['ruta_video']}")
        
        self.ancho = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        
        # Historial de datos
        self.datos_frames: List[Dict[str, Any]] = []
        self.total_frames = 0

    def punto_en_poligono(self, punto, poligono):
        """Determina si un punto está dentro de un polígono"""
        return cv2.pointPolygonTest(poligono, punto, False) >= 0

    def detectar_vehiculos(self, frame: np.ndarray) -> List[Dict]:
        """Detecta vehículos usando YOLOv11"""
        detecciones = []
        resultados = self.modelo.predict(
            frame, 
            conf=self.configuracion["umbral_confianza"], 
            classes=self.configuracion["clases_vehiculos"],
            verbose=False
        )
        
        for caja in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            detecciones.append({
                'caja': (x1, y1, x2, y2),
                'centro': centro,
                'confianza': float(caja.conf[0]),
                'clase': int(caja.cls[0])
            })
        
        return detecciones

    def contar_vehiculos_por_zona(self, detecciones: List[Dict]) -> Dict[str, int]:
        """Cuenta vehículos en cada zona de interés"""
        conteo = {zona: 0 for zona in self.configuracion["zonas_interes"]}
        
        for det in detecciones:
            for zona, poligono in self.configuracion["zonas_interes"].items():
                if self.punto_en_poligono(det['centro'], poligono):
                    conteo[zona] += 1
                    break
        
        return conteo

    def dibujar_resultados(self, frame: np.ndarray, detecciones: List[Dict], conteo_zonas: Dict[str, int]) -> np.ndarray:
        """Dibuja las zonas, detecciones y conteo en el frame"""
        frame_anotado = frame.copy()
        
        # Dibujar zonas de interés con el nuevo grosor
        for zona, poligono in self.configuracion["zonas_interes"].items():
            color = self.configuracion["colores_zonas"][zona]
            grosor_zona = self.configuracion["estilo_zonas"]["grosor"]
            cv2.polylines(frame_anotado, [poligono], True, color, grosor_zona)
            cv2.putText(frame_anotado, zona, tuple(poligono[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar detecciones de vehículos con nuevos estilos
        color_bbox = self.configuracion["estilo_bounding_box"]["color"]
        grosor_bbox = self.configuracion["estilo_bounding_box"]["grosor"]
        color_texto = self.configuracion["estilo_texto_confianza"]["color"]
        grosor_texto = self.configuracion["estilo_texto_confianza"]["grosor"]
        
        for det in detecciones:
            x1, y1, x2, y2 = det['caja']
            # Dibujar bounding box con nuevo color y grosor
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color_bbox, grosor_bbox)
            # Dibujar texto de confianza con nuevo color y grosor
            cv2.putText(frame_anotado, f"{det['confianza']:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, grosor_texto)
        
        # Mostrar conteo por zona
        for i, (zona, count) in enumerate(conteo_zonas.items()):
            color = self.configuracion["colores_zonas"][zona]
            cv2.putText(frame_anotado, f"{zona}: {count}", (10, 30 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Mostrar total de vehículos detectados
        total_vehiculos = sum(conteo_zonas.values())
        cv2.putText(frame_anotado, f"Total: {total_vehiculos}", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_anotado

    def procesar_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame completo"""
        self.total_frames += 1
        
        # Detectar vehículos
        detecciones = self.detectar_vehiculos(frame)
        
        # Contar por zonas
        conteo_zonas = self.contar_vehiculos_por_zona(detecciones)
        
        # Almacenar datos
        self.datos_frames.append({
            'frame': self.total_frames,
            'tiempo': self.total_frames / self.fps,
            'total_vehiculos': sum(conteo_zonas.values()),
            **conteo_zonas
        })
        
        # Dibujar resultados
        frame_anotado = self.dibujar_resultados(frame, detecciones, conteo_zonas)
        
        return frame_anotado

    def ejecutar_analisis(self):
        """Ejecuta el análisis completo del video"""
        print("\nIniciando análisis de densidad de vehículos... (Presione 'q' para salir)")
        cv2.namedWindow("Densidad de Vehículos", cv2.WINDOW_NORMAL)
        
        while self.captura.isOpened():
            ret, frame = self.captura.read()
            if not ret:
                break
            
            frame_procesado = self.procesar_frame(frame)
            cv2.imshow("Densidad de Vehículos", frame_procesado)
            
            # Control para salir
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self._limpiar_recursos()
        self._mostrar_estadisticas()

    def _mostrar_estadisticas(self):
        """Muestra estadísticas finales del análisis"""
        if not self.datos_frames:
            return
        
        print("\n=== ESTADÍSTICAS FINALES ===")
        print(f"Total de frames procesados: {self.total_frames}")
        print(f"Duración del video: {self.total_frames / self.fps:.2f} segundos")
        
        # Calcular estadísticas por zona
        for zona in self.configuracion["zonas_interes"].keys():
            valores = [frame[zona] for frame in self.datos_frames]
            promedio = sum(valores) / len(valores)
            maximo = max(valores)
            print(f"{zona}: Promedio={promedio:.2f}, Máximo={maximo}")

    def guardar_resultados(self, ruta_salida: Path):
        """Guarda los resultados del análisis"""
        import pandas as pd
        
        ruta_salida.mkdir(parents=True, exist_ok=True)
        ruta_csv = ruta_salida / 'densidad_vehiculos.csv'
        
        df = pd.DataFrame(self.datos_frames)
        df.to_csv(ruta_csv, index=False)
        print(f"\nDatos de densidad guardados en: {ruta_csv}")

    def _limpiar_recursos(self):
        """Libera recursos de video"""
        self.captura.release()
        cv2.destroyAllWindows()


def principal():
    """Función principal"""
    try:
        contador = ContadorDensidadVehiculos(CONFIGURACION)
        contador.ejecutar_analisis()
        
        # Guardar resultados (opcional)
        # contador.guardar_resultados(Path('/ruta/para/resultados'))
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    principal()