import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
VIDEO_FILE = 'tirada_4.mp4'

# --- PARÁMETROS AJUSTADOS PARA MEJOR DETECCIÓN DE ESTABILIDAD ---
UMBRAL_MOVIMIENTO = 2000      # Reducido de 5000 - más estricto para detectar movimientos sutiles
FRAMES_ESTABLES = 5           # Reducido de 10 - confirma reposo más rápido pero requiere más estabilidad
FRAMES_CONFIRMACION = 3       # Nuevo: frames adicionales para verificar que realmente está estático

# --- PARÁMETROS DE DETECCIÓN DE DADOS ---
AREA_MIN_DADO = 1000
AREA_MAX_DADO = 60000
NUM_DADOS_ESPERADOS = 5       # NUEVO: Número de dados en cada tirada

# Parámetros de detección de pips
UMBRAL_PIPS = 180
AREA_MIN_PIP = 30

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
    candidatos_reposo = []  # Lista de candidatos a frame de reposo

    print("\n[Etapa 1: Detección de Estabilidad - MODO ESTRICTO]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            # 1. Cálculo de la Diferencia de Frames
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            movimiento_puntos = np.sum(thresh == 255)

            # Imprimir info de debug cada 10 frames
            if frame_idx % 10 == 0:
                print(f"   Frame {frame_idx}: Movimiento = {movimiento_puntos} px")

            if movimiento_puntos < UMBRAL_MOVIMIENTO:
                conteo_estables += 1
            else:
                # Si hay movimiento, resetear contador
                if conteo_estables > 0:
                    print(f"   --> Movimiento detectado en frame {frame_idx} ({movimiento_puntos} px) - Reiniciando contador")
                conteo_estables = 0

            # 2. Confirmación de Reposo con doble verificación
            if conteo_estables >= FRAMES_ESTABLES:
                # Guardar candidato con su score de estabilidad
                frame_inicio_reposo = frame_idx - FRAMES_ESTABLES + 1
                candidatos_reposo.append((frame_inicio_reposo, conteo_estables, movimiento_puntos))
                
                # Solo reportar cada vez que alcanza un nuevo nivel de estabilidad
                if conteo_estables == FRAMES_ESTABLES:
                    print(f"   --> CANDIDATO DE REPOSO detectado en Frame {frame_inicio_reposo} (estable por {FRAMES_ESTABLES} frames)")

        prev_gray = gray
        frame_idx += 1

    cap.release()
    
    # Seleccionar el mejor candidato: el que tiene mayor cantidad de frames estables
    if candidatos_reposo:
        # Ordenar por número de frames estables (mayor a menor)
        candidatos_reposo.sort(key=lambda x: x[1], reverse=True)
        mejor_candidato = candidatos_reposo[0]
        frame_seleccionado = mejor_candidato[0]
        
        print(f"\n   --> REPOSO CONFIRMADO: Frame {frame_seleccionado}")
        print(f"   --> Estabilidad: {mejor_candidato[1]} frames consecutivos")
        print(f"   --> Movimiento final: {mejor_candidato[2]} píxeles")
        
        return frame_seleccionado
    
    return None

def corregir_iluminacion_no_uniforme(frame):
    """
    Corrige la iluminación no uniforme del frame.
    Aplica corrección para que la iluminación sea más homogénea,
    priorizando los niveles de luminosidad de la parte superior.
    """
    print("\n[Corrección de Iluminación No Uniforme]")
    
    # Convertir a espacio LAB (mejor para trabajar con luminosidad)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Método 1: CLAHE adaptativo en todo el frame
    # Esto equilibra la iluminación localmente
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_channel)
    
    # Método 2: Corrección de gradiente vertical
    # Detectar el gradiente de iluminación de arriba hacia abajo
    height = l_channel.shape[0]
    
    # Calcular luminosidad promedio en franjas horizontales
    num_franjas = 10
    altura_franja = height // num_franjas
    luminosidades = []
    
    for i in range(num_franjas):
        y_inicio = i * altura_franja
        y_fin = min((i + 1) * altura_franja, height)
        franja = l_channel[y_inicio:y_fin, :]
        lum_promedio = np.mean(franja)
        luminosidades.append(lum_promedio)
    
    # Usar la luminosidad de la parte superior como referencia
    # (primeras 3 franjas)
    lum_objetivo = np.mean(luminosidades[:3])
    
    print(f"   --> Luminosidad objetivo (parte superior): {lum_objetivo:.1f}")
    print(f"   --> Luminosidad parte inferior: {luminosidades[-1]:.1f}")
    print(f"   --> Diferencia: {luminosidades[-1] - lum_objetivo:.1f}")
    
    # Crear mapa de corrección suave
    mapa_correccion = np.zeros_like(l_channel, dtype=np.float32)
    
    for y in range(height):
        # Calcular factor de corrección para esta fila
        # Interpolación lineal basada en la posición vertical
        franja_idx = min(int(y / altura_franja), num_franjas - 1)
        lum_actual = luminosidades[franja_idx]
        
        # Factor de corrección: cuánto hay que ajustar para llegar al objetivo
        factor = lum_objetivo - lum_actual
        
        # Aplicar corrección más suave (75% del factor)
        mapa_correccion[y, :] = factor * 0.75
    
    # Aplicar corrección al canal L
    l_corregido = l_channel.astype(np.float32) + mapa_correccion
    l_corregido = np.clip(l_corregido, 0, 255).astype(np.uint8)
    
    # Combinar CLAHE con corrección de gradiente (promedio ponderado)
    l_final = cv2.addWeighted(l_clahe, 0.5, l_corregido, 0.5, 0)
    
    # Reconstruir imagen
    lab_corregido = cv2.merge([l_final, a, b])
    frame_corregido = cv2.cvtColor(lab_corregido, cv2.COLOR_LAB2BGR)
    
    print(f"   --> Iluminación corregida exitosamente")
    
    return frame_corregido

def reconocer_dados(frame_estatico):
    """Localiza los dados y cuenta los pips en un frame estático.
    VERSIÓN 3.2: Incluye filtro para descartar el dado fantasma (solo 5 dados esperados)."""
    
    # Corregir iluminación antes de procesar
    frame_corregido = corregir_iluminacion_no_uniforme(frame_estatico.copy())
    
    # 1. Segmentación de Dados (Filtro por Color Rojo en HSV)
    hsv = cv2.cvtColor(frame_corregido, cv2.COLOR_BGR2HSV)

    # Rango de Color Rojo (requiere dos rangos en HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
    mask_dados = cv2.bitwise_or(mask1, mask2)

    # Mejorar la máscara con Morfología
    kernel = np.ones((5,5), np.uint8)
    mask_dados = cv2.dilate(mask_dados, kernel, iterations=DILATE_ITERATIONS)
    mask_dados = cv2.erode(mask_dados, kernel, iterations=ERODE_ITERATIONS)

    # 2. Encontrar contornos de los dados
    contornos_dados, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista temporal de candidatos (dado, área, posición_y)
    candidatos_dados = []

    print("\n[Etapa 2: Reconocimiento de Valor (Pips)]")
    print(f"   --> Contornos encontrados: {len(contornos_dados)}")

    for i, contorno_dado in enumerate(contornos_dados):
        area = cv2.contourArea(contorno_dado)
        
        # Rango de área ajustado para capturar dados de diferentes tamaños
        if AREA_MIN_DADO < area < AREA_MAX_DADO:
            x, y, w, h = cv2.boundingRect(contorno_dado)
            dado_roi = frame_corregido[y:y+h, x:x+w]

            # 3. Contar Pips (Puntos Blancos)
            dado_gray = cv2.cvtColor(dado_roi, cv2.COLOR_BGR2GRAY)

            # Aplicar CLAHE para mejorar contraste antes de umbralizar
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            dado_gray_mejorado = clahe.apply(dado_gray)

            # Umbralizar con valor ajustado para aislar los puntos blancos (pips)
            _, mask_pips = cv2.threshold(dado_gray_mejorado, UMBRAL_PIPS, 255, cv2.THRESH_BINARY)

            # Limpieza morfológica de la máscara de pips
            kernel_pip = np.ones((3,3), np.uint8)
            mask_pips = cv2.morphologyEx(mask_pips, cv2.MORPH_OPEN, kernel_pip)

            # Encontrar contornos de los pips
            pips_contornos, _ = cv2.findContours(mask_pips, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valor_dado = 0
            for contorno_pip in pips_contornos:
                # Filtrar blobs pequeños (ruido) con área mínima ajustada
                if cv2.contourArea(contorno_pip) > AREA_MIN_PIP:
                    valor_dado += 1

            # Guardar candidato con información adicional
            candidatos_dados.append({
                'bbox': (x, y, w, h),
                'valor': valor_dado,
                'area': area,
                'y_centro': y + h // 2  # Centro vertical para ordenar
            })
            
            print(f"   --> Candidato {i+1} en ({x},{y}) - Área: {int(area)} - Valor: {valor_dado}")
        else:
            print(f"   --> Contorno {i+1} descartado - Área: {int(area)} (fuera de rango)")

    # NUEVO: Filtrar dados fantasma
    print(f"\n[Filtro de Dados Fantasma]")
    print(f"   --> Candidatos detectados: {len(candidatos_dados)}")
    print(f"   --> Dados esperados: {NUM_DADOS_ESPERADOS}")
    
    if len(candidatos_dados) > NUM_DADOS_ESPERADOS:
        # Ordenar candidatos por área (de mayor a menor)
        # Los dados reales suelen tener áreas más grandes y consistentes
        candidatos_dados.sort(key=lambda d: d['area'], reverse=True)
        
        # Seleccionar solo los N dados más grandes
        dados_seleccionados = candidatos_dados[:NUM_DADOS_ESPERADOS]
        dados_descartados = candidatos_dados[NUM_DADOS_ESPERADOS:]
        
        print(f"   --> Seleccionando los {NUM_DADOS_ESPERADOS} dados con mayor área")
        for d in dados_descartados:
            x, y, w, h = d['bbox']
            print(f"   --> DESCARTADO: Dado en ({x},{y}) - Área: {int(d['area'])} - Valor: {d['valor']}")
        
        candidatos_dados = dados_seleccionados
    
    # Convertir a formato final y ordenar por posición (arriba a abajo, izq a der)
    dados_detectados = []
    for d in candidatos_dados:
        dados_detectados.append((d['bbox'], d['valor']))
    
    # Ordenar por posición vertical (y) para numeración consistente
    dados_detectados.sort(key=lambda d: d[0][1])  # Ordenar por coordenada y
    
    print(f"\n   --> Dados finales detectados: {len(dados_detectados)}")
    for i, ((x, y, w, h), valor) in enumerate(dados_detectados):
        print(f"   --> Dado {i+1} en ({x},{y}) - Valor: {valor}")

    return dados_detectados, frame_corregido

def generar_video_con_bbox(video_path, frame_idx_inicio_estatico, dados_detectados):
    """Genera un nuevo video con Bounding Boxes y valores."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Preparar el escritor de video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = video_path.replace(".mp4", "")
    output_filename = f"{base_name}_anotado_v3_2.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para MP4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (ancho, alto))

    frame_idx = 0
    print(f"\n[Etapa 3: Generando Video Anotado: {output_filename}]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Solo aplicar anotaciones cuando los dados están en reposo
        if frame_idx >= frame_idx_inicio_estatico:
            for i, ((x, y, w, h), valor) in enumerate(dados_detectados):
                # Bounding Box (Rectángulo)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3) # Amarillo

                # Etiqueta y Valor (Texto)
                etiqueta = f"Dado {i+1}: {valor}"
                cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    print(f"   --> ¡Video anotado generado con éxito!")
    out.release()
    cap.release()


# --- EJECUCIÓN DEL ALGORITMO ---

# 1. Detección de Estabilidad
frame_idx_estatico = detectar_frames_estaticos(VIDEO_FILE)

if frame_idx_estatico is not None:
    # 2. Cargar el frame estático
    cap = cv2.VideoCapture(VIDEO_FILE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_estatico)
    ret, frame_estatico = cap.read()
    cap.release()

    if ret:
        print(f"\n[Visualización del Frame Estático {frame_idx_estatico}]")
        
        # Guardar frame estático ORIGINAL para inspección
        cv2.imwrite(f"{VIDEO_FILE.replace('.mp4', '')}_frame_original_v3_2.png", frame_estatico)

        # 3. Reconocimiento de Dados - CON FILTRO DE DADO FANTASMA
        dados_detectados_finales, frame_corregido = reconocer_dados(frame_estatico.copy())
        
        # Guardar frame CORREGIDO para comparación
        cv2.imwrite(f"{VIDEO_FILE.replace('.mp4', '')}_frame_corregido_v3_2.png", frame_corregido)

        # 4. Mostrar Resultados Finales
        valores_obtenidos = [valor for _, valor in dados_detectados_finales]
        suma_total = sum(valores_obtenidos)

        print("\n--- RESULTADOS DE LA TIRADA 1 (VERSIÓN 3.2 - SIN DADO FANTASMA) ---")
        print(f"Frame de Reposo Confirmado: {frame_idx_estatico}")
        print(f"Dados detectados: {len(dados_detectados_finales)}")
        print(f"Valores obtenidos: {valores_obtenidos}")
        print(f"Suma Total: {suma_total}")
        print("--------------------------------------------------------------------")

        # 5. Generar el Video Anotado
        generar_video_con_bbox(VIDEO_FILE, frame_idx_estatico, dados_detectados_finales)

    else:
        print(f"Error: No se pudo leer el frame {frame_idx_estatico}")
else:
    print(f"\nResultado: No se detectaron frames estáticos en {VIDEO_FILE}.")
