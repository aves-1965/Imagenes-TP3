import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
VIDEO_FILE = 'tirada_4.mp4'

# --- PARÁMETROS DE DETECCIÓN DE ESTABILIDAD ---
UMBRAL_MOVIMIENTO = 2000
FRAMES_ESTABLES = 5
FRAMES_CONFIRMACION = 3

# --- PARÁMETROS DE DETECCIÓN DE DADOS ---
AREA_MIN_DADO = 1000
AREA_MAX_DADO = 60000

# --- PARÁMETROS MEJORADOS PARA DETECCIÓN DE PIPS BLANCOS ---
# Umbrales para detección de blanco en espacio LAB
L_MIN = 180          # Luminosidad mínima (blanco brillante)
L_MAX = 255
A_MIN = 0            # Canal a* (verde-rojo) cerca de neutro
A_MAX = 135
B_MIN = 0            # Canal b* (azul-amarillo) cerca de neutro  
B_MAX = 135

# Umbrales en HSV como respaldo
SAT_MAX = 80         # Saturación máxima para blanco (muy baja)
VAL_MIN = 200        # Valor mínimo (muy brillante)

# Parámetros de forma
CIRCULARIDAD_MIN = 0.75    # Mínima circularidad (1.0 = círculo perfecto)
AREA_MIN_PIP = 40         # Área mínima de pip
AREA_MAX_PIP = 800        # Área máxima de pip (evita áreas grandes)

# Parámetros de morfología
DILATE_ITERATIONS = 3
ERODE_ITERATIONS = 2

# --- FUNCIONES DE PROCESAMIENTO ---

def detectar_frames_estaticos(video_path):
    """Detecta el primer índice de frame donde los dados se detienen completamente.
    VERSIÓN MEJORADA: Detección más estricta con doble verificación."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return None

    prev_gray = None
    conteo_estables = 0
    frame_idx = 0
    candidatos_reposo = []

    print("\n[Etapa 1: Detección de Estabilidad - MODO ESTRICTO]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            movimiento_puntos = np.sum(thresh == 255)

            if frame_idx % 10 == 0:
                print(f"   Frame {frame_idx}: Movimiento = {movimiento_puntos} px")

            if movimiento_puntos < UMBRAL_MOVIMIENTO:
                conteo_estables += 1
            else:
                if conteo_estables > 0:
                    print(f"   --> Movimiento detectado en frame {frame_idx} ({movimiento_puntos} px) - Reiniciando contador")
                conteo_estables = 0

            if conteo_estables >= FRAMES_ESTABLES:
                frame_inicio_reposo = frame_idx - FRAMES_ESTABLES + 1
                candidatos_reposo.append((frame_inicio_reposo, conteo_estables, movimiento_puntos))
                
                if conteo_estables == FRAMES_ESTABLES:
                    print(f"   --> CANDIDATO DE REPOSO detectado en Frame {frame_inicio_reposo} (estable por {FRAMES_ESTABLES} frames)")

        prev_gray = gray
        frame_idx += 1

    cap.release()
    
    if candidatos_reposo:
        candidatos_reposo.sort(key=lambda x: x[1], reverse=True)
        mejor_candidato = candidatos_reposo[0]
        frame_seleccionado = mejor_candidato[0]
        
        print(f"\n   --> REPOSO CONFIRMADO: Frame {frame_seleccionado}")
        print(f"   --> Estabilidad: {mejor_candidato[1]} frames consecutivos")
        print(f"   --> Movimiento final: {mejor_candidato[2]} píxeles")
        
        return frame_seleccionado
    
    return None

def detectar_pips_por_color_blanco(dado_roi):
    """
    Detecta pips usando detección de blanco puro en múltiples espacios de color.
    Filtra por forma circular y brillo para evitar detectar caras transparentes.
    """
    
    # 1. DETECCIÓN EN ESPACIO LAB (mejor para blancos)
    lab = cv2.cvtColor(dado_roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Máscara de blanco en LAB
    mask_lab = cv2.inRange(lab, 
                          np.array([L_MIN, A_MIN, B_MIN]), 
                          np.array([L_MAX, A_MAX, B_MAX]))
    
    # 2. DETECCIÓN EN ESPACIO HSV (saturación baja)
    hsv = cv2.cvtColor(dado_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Máscara de blanco en HSV (saturación muy baja, valor alto)
    mask_hsv = cv2.inRange(hsv,
                          np.array([0, 0, VAL_MIN]),
                          np.array([180, SAT_MAX, 255]))
    
    # 3. COMBINAR MÁSCARAS (AND lógico - debe cumplir ambas condiciones)
    mask_blanco = cv2.bitwise_and(mask_lab, mask_hsv)
    
    # 4. LIMPIEZA MORFOLÓGICA
    kernel_small = np.ones((2,2), np.uint8)
    mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_OPEN, kernel_small)
    mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, kernel_small)
    
    # 5. ENCONTRAR CONTORNOS DE PIPS
    contornos_pips, _ = cv2.findContours(mask_blanco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. FILTRAR POR ÁREA, CIRCULARIDAD Y BRILLO
    pips_validos = []
    
    for contorno in contornos_pips:
        area = cv2.contourArea(contorno)
        
        # Filtro por área
        if AREA_MIN_PIP < area < AREA_MAX_PIP:
            # Calcular circularidad: 4π*área / perímetro²
            perimetro = cv2.arcLength(contorno, True)
            if perimetro == 0:
                continue
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            
            # Filtro por circularidad (los pips son circulares)
            if circularidad >= CIRCULARIDAD_MIN:
                # Obtener región del pip para analizar brillo
                x, y, w, h = cv2.boundingRect(contorno)
                pip_region = l_channel[y:y+h, x:x+w]
                
                # Calcular brillo promedio
                brillo_promedio = np.mean(pip_region)
                
                # Los pips de la cara superior son más brillantes
                # que los transparentados de la cara inferior
                if brillo_promedio >= L_MIN:
                    pips_validos.append({
                        'contorno': contorno,
                        'area': area,
                        'circularidad': circularidad,
                        'brillo': brillo_promedio,
                        'bbox': (x, y, w, h)
                    })
    
    # 7. SELECCIONAR LOS PIPS MÁS BRILLANTES (máximo 6)
    # Ordenar por brillo descendente
    pips_validos.sort(key=lambda p: p['brillo'], reverse=True)
    
    # Si hay más de 6, tomar solo los 6 más brillantes
    if len(pips_validos) > 6:
        print(f"      [!] {len(pips_validos)} candidatos detectados, seleccionando los 6 más brillantes")
        pips_validos = pips_validos[:6]
    
    return len(pips_validos), pips_validos

def reconocer_dados(frame_estatico):
    """Localiza los dados y cuenta los pips en un frame estático.
    VERSIÓN V4: Detección mejorada por color blanco y circularidad."""
    
    # 1. Segmentación de Dados (Filtro por Color Rojo en HSV)
    hsv = cv2.cvtColor(frame_estatico, cv2.COLOR_BGR2HSV)

    # Rango de Color Rojo
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
    mask_dados = cv2.bitwise_or(mask1, mask2)

    # Mejorar la máscara con Morfología
    kernel = np.ones((5,5), np.uint8)
    mask_dados = cv2.dilate(mask_dados, kernel, iterations=DILATE_ITERATIONS)
    mask_dados = cv2.erode(mask_dados, kernel, iterations=ERODE_ITERATIONS)

    # 2. Encontrar contornos de los dados
    contornos_dados, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dados_detectados = []

    print("\n[Etapa 2: Reconocimiento de Valor (Pips) - DETECCIÓN POR COLOR BLANCO]")
    print(f"   --> Contornos encontrados: {len(contornos_dados)}")

    for i, contorno_dado in enumerate(contornos_dados):
        area = cv2.contourArea(contorno_dado)
        
        if AREA_MIN_DADO < area < AREA_MAX_DADO:
            x, y, w, h = cv2.boundingRect(contorno_dado)
            dado_roi = frame_estatico[y:y+h, x:x+w]

            # 3. CONTAR PIPS USANDO DETECCIÓN DE BLANCO MEJORADA
            valor_dado, pips_info = detectar_pips_por_color_blanco(dado_roi)

            # Información detallada para debug
            if pips_info:
                brillos = [p['brillo'] for p in pips_info]
                circularidades = [p['circularidad'] for p in pips_info]
                print(f"   --> Dado {i+1} en ({x},{y}) - Área: {int(area)}")
                print(f"       Pips detectados: {valor_dado}")
                print(f"       Brillo promedio pips: {np.mean(brillos):.1f}")
                print(f"       Circularidad promedio: {np.mean(circularidades):.2f}")
            else:
                print(f"   --> Dado {i+1} en ({x},{y}) - Área: {int(area)} - Valor: {valor_dado}")

            dados_detectados.append(((x, y, w, h), valor_dado))
        else:
            print(f"   --> Contorno {i+1} descartado - Área: {int(area)} (fuera de rango)")

    return dados_detectados

def generar_video_con_bbox(video_path, frame_idx_inicio_estatico, dados_detectados):
    """Genera un nuevo video con Bounding Boxes y valores."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = video_path.replace(".mp4", "")
    output_filename = f"{base_name}_anotado_v4.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (ancho, alto))

    frame_idx = 0
    print(f"\n[Etapa 3: Generando Video Anotado: {output_filename}]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= frame_idx_inicio_estatico:
            for i, ((x, y, w, h), valor) in enumerate(dados_detectados):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                etiqueta = f"Dado {i+1}: {valor}"
                cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    print(f"   --> ¡Video anotado generado con éxito!")
    out.release()
    cap.release()


# --- EJECUCIÓN DEL ALGORITMO ---

frame_idx_estatico = detectar_frames_estaticos(VIDEO_FILE)

if frame_idx_estatico is not None:
    cap = cv2.VideoCapture(VIDEO_FILE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_estatico)
    ret, frame_estatico = cap.read()
    cap.release()

    if ret:
        print(f"\n[Visualización del Frame Estático {frame_idx_estatico}]")
        cv2.imwrite(f"{VIDEO_FILE.replace('.mp4', '')}_frame_estatico_v4.png", frame_estatico)

        dados_detectados_finales = reconocer_dados(frame_estatico.copy())

        valores_obtenidos = [valor for _, valor in dados_detectados_finales]
        suma_total = sum(valores_obtenidos)

        print("\n--- RESULTADOS TIRADA 1 (V4 - DETECCIÓN BLANCO MEJORADA) ---")
        print(f"Frame de Reposo: {frame_idx_estatico}")
        print(f"Dados detectados: {len(dados_detectados_finales)}")
        print(f"Valores obtenidos: {valores_obtenidos}")
        print(f"Suma Total: {suma_total}")
        print("-------------------------------------------------------------")

        generar_video_con_bbox(VIDEO_FILE, frame_idx_estatico, dados_detectados_finales)

    else:
        print(f"Error: No se pudo leer el frame {frame_idx_estatico}")
else:
    print(f"\nResultado: No se detectaron frames estáticos en {VIDEO_FILE}.")
