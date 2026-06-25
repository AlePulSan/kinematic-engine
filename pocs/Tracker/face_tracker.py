import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

def extract_face_mask(frame):
    """
    Este método toma un frame, extrae los puntos de la cara
    y devuelve una máscara binaria (blanco y negro) de la silueta.
    """
    h, w, _ = frame.shape

    # OpenCV usa BGR por defecto, pero las redes neuronales suelen pedir RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = mp_face_mesh.process(frame_rgb)
    
    # Se crea una ventana negra del mismo tamaño que la cámara
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # procesar
    if results.multi_face_landmarks:
        # Cogemos la primera (y en este caso única max_num_faces=1) cara detectada
        face_landmarks = results.multi_face_landmarks[0]
        
        # Mapeo de coordenadas
        # MediaPipe devuelve % (de 0.0 a 1.0). Multiplicamos por la resolución real.
        # Forzamos np.int32 porque OpenCV C++ colapsa si le pasas floats a una función de dibujo.
        pts = np.array([
            (int(landmark.x * w), int(landmark.y * h)) 
            for landmark in face_landmarks.landmark
        ], dtype=np.int32)
        
        # Convex Hull se encarga de calcular el polígono exterior que envuelve todos esos puntos.
        hull = cv2.convexHull(pts)
        
        # Rellena ese polígono de blanco (255) en nuestro lienzo negro.
        cv2.fillConvexPoly(mask, hull, 255)
        
    return mask

if __name__ == "__main__":
    print("[Test] Arrancando extracción de telemetría facial...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Error] No se detecta cámara.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Invertimos la imagen
            frame = cv2.flip(frame, 1)
            
            # Obtenemos la máscara de la función
            mask = extract_face_mask(frame)
            
            # Mostramos el frame original y la máscara al lado para comparar
            cv2.imshow("Original", frame)
            cv2.imshow("Mascara (Convex Hull)", mask)
            
            if cv2.waitKey(1) & 0xFF == 27: #Salida del programa con ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Info] Limpieza terminada.")