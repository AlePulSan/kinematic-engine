import cv2
import time
from threading import Thread
from queue import Queue

class ThreadedCamera:
    def __init__(self, src=0, queue_size=128):
        self.cap = cv2.VideoCapture(src)
        self.is_file = isinstance(src, str) # Diferenciamos si es archivo (path) o webcam (número)
        
        # Si es archivo, usamos una cola (Queue) para no saltarnos ningún frame. 
        # Si es webcam, no usamos cola para evitar que las fotos se acumulen y haya lag.
        self.q = Queue(maxsize=queue_size) if self.is_file else None
        
        if not self.is_file:
            self.ret, self.frame = self.cap.read() # Disparamos la primera foto si es webcam
            
        self.stopped = False

    def start(self):
        # Arranca el bucle de lectura en segundo plano con un hilo
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        # Bucle secundario no para de pedirle fotos a la cámara
        while not self.stopped:
            if self.is_file:
                # MODO ARCHIVO: Llenamos la cola ordenadamente
                if not self.q.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stop() # Se acabó el video
                        break
                    self.q.put(frame)
                else:
                    time.sleep(0.005) # esperamos si la cola está llena
            else:
                # MODO WEBCAM: Se sustituye la foto vieja con la más nueva.
                self.ret, self.frame = self.cap.read()
                time.sleep(0.001)

    def read(self):
        # El programa principal llama a este metodo para pedir la foto
        if self.is_file:
            if not self.q.empty():
                return True, self.q.get() # Saca la foto más antigua de la cola
            return False, None
        else:
            return self.ret, self.frame # Entrega la foto

    def stop(self):
        self.stopped = True
        self.cap.release()

# test
if __name__ == "__main__":
    print("[Test] Arrancando lector de video multihilo...")
    
    video_source = 0 # 0 para webcam o "ruta/al/video.mp4"
    cam = ThreadedCamera(src=video_source).start()
    
    # Check de seguridad básico
    if not cam.cap.isOpened():
        print("[Error] No hay cámara o la ruta del video está mal.")
        cam.stop()
        exit()
        
    try:
        while True:
            time.sleep(0.03) # Simulamos que el programa está ocupado
            
            ret, frame = cam.read()
            
            if not ret or frame is None:
                if cam.stopped:
                    print("[Info] Fin del video o se desconectó la cámara.")
                    break
                continue 
            
            cv2.imshow("Prueba Camara Paralela", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # Salir tecla ESC
                break
                
    except KeyboardInterrupt:
        print("\n[Info] Cierre forzado con Ctrl+C.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[Info] Limpieza terminada.")