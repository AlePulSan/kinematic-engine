import cv2
import torch
import torch.nn.functional as F
import numpy as np

def main():
    # Buscamos 'cuda' (tarjetas Nvidia). Si no hay, hacemos fallback a procesador ('cpu').
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Render asignado a: {device}")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("No se detecta cámara.")
        return

    H, W, _ = frame.shape

    # PyTorch no usa píxeles (0 a 1280). Usa un plano de coordenadas normalizado de -1.0 a 1.0.
    # Generamos un tensor (matriz de VRAM) con las coordenadas X e Y fijas de cada píxel de la pantalla.
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), 
                          torch.linspace(-1, 1, W, device=device), 
                          indexing='ij')
                          
    # Apilamos X e Y para tener coordenadas (x, y) por cada punto. 
    base_grid = torch.stack((x, y), dim=2).unsqueeze(0)

    t = 0  #tiempo
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Pasamos de matriz NumPy (RAM) a Tensor PyTorch (VRAM). 
            # Normalizamos el color dividiendo por 255.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(frame_rgb).to(device).float() / 255.0
            
            # PyTorch exige estructura: [Batch, Canales, Alto, Ancho]. Reordenamos la memoria.
            frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)

            # calculamos vectores
            t += 0.1
            # Simulamos el efecto deformación sumando una onda (seno y coseno) a los ejes
            desplazamiento_x = torch.sin(y * 10 + t) * 0.05
            desplazamiento_y = torch.cos(x * 10 + t) * 0.05
            
            # Creamos la nueva malla diciendo a dónde debe ir cada píxel
            distorsion = torch.stack((desplazamiento_x, desplazamiento_y), dim=2).unsqueeze(0)
            grid_distorsionado = base_grid + distorsion

            # remapeamos
            # F.grid_sample coge la imagen original (frame_t) y lee el mapa de coordenadas (grid_distorsionado)
            # Hace el traslado de memoria
            out_t = F.grid_sample(frame_t, grid_distorsionado, mode='bilinear', padding_mode='border', align_corners=True)

            # quitamos el Batch, volvemos a formato [Alto, Ancho, Canales],
            # bajamos a la CPU, lo convertimos a NumPy, escalamos a 255 y pasamos a enteros (uint8).
            out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

            cv2.imshow("Motor de Remapeo GPU", out)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Limpieza de memoria hecha")

if __name__ == "__main__":
    main()