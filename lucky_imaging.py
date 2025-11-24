#Python 3.11
import os
import numpy as np
from astropy.io import fits
from scipy import ndimage
from skimage.restoration import richardson_lucy
from scipy.ndimage import gaussian_filter

def lucky_imaging(input_data, params=None):
    # ConfiguraciÃ³n 
    default_params = {
        "selection_method": "peak",                
        "selection_percentage": 10,
        "alignment_type": "robust",
        "alignment_method": "cross_correlation",        
        "stacking_method": "mean",                 
        "apply_deconvolution": False        
    }

    if params is None:
        params = default_params
    else:
        for k, v in default_params.items():
            params.setdefault(k, v)

    # Carga de datos
    if isinstance(input_data, str):
        frames = load_fits_frames(input_data)
    else:
        frames = input_data

    if frames.ndim != 3:
        raise ValueError("Input debe ser un volumen 3D (H, W, N).")

    # 1. Medicion de calidad y Ranking
    ranked_indices, quality_scores = rank_frames(frames, method=params["selection_method"])

    # 2. Seleccion de frames
    selected_indices = select_best_frames(ranked_indices, params["selection_percentage"])
    selected_frames = frames[:, :, selected_indices]
    selected_weights = quality_scores[selected_indices]
    # 2.5 Guardar mejor frame
    best_frame_raw = selected_frames[:, :, 0]

    # 3. Alineamiento
    reference_frame = selected_frames[:, :, 0]
    
    if params.get("alignment", True):
        if params["alignment_type"] == "robust":
            aligned_frames = align_frames_robust(selected_frames, reference_frame, sigma_smooth=2.0)
        else:
            aligned_frames = align_frames_standard(selected_frames, reference_frame)
    else:
        aligned_frames = selected_frames

    # 4. Apilamiento
    stacked_image = stack_images(aligned_frames, params["stacking_method"], selected_weights)
    
    # 5. Mejora (DeconvoluciÃ³n)
    if params["apply_deconvolution"]:
        stacked_image = apply_safe_deconvolution(stacked_image)

    # Generar imagen base (promedio simple) para comparar
    baseline_image = np.mean(frames, axis=2)

    result = {
        "image": stacked_image,
        "baseline": baseline_image,
        "best_frame": best_frame_raw,
        "params": params,
        "selected_count": len(selected_indices)
    }

    return result

def calculate_spectral_quality(frame):
    """
    Calcula la calidad de la imagen basÃ¡ndose en la energÃ­a de alta frecuencia usando la Transformada de Fourier (FFT).
    """
    # Transformada de Fourier
    f_transform = np.fft.fft2(frame)
    f_shift = np.fft.fftshift(f_transform)
    
    # Espectro de potencia
    power_spectrum = np.abs(f_shift)**2
    
    # mascara + filtrado low frec + ruido high frec = detalle medio alto
    rows, cols = frame.shape
    crow, ccol = rows // 2, cols // 2
    
    # Radio interno
    r_inner = int(min(rows, cols) * 0.05) 
    # Radio externo 
    r_outer = int(min(rows, cols) * 0.5)
    
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2
    mask = (mask_area >= r_inner**2) & (mask_area <= r_outer**2)
    
    quality_score = np.sum(power_spectrum[mask])
    
    return quality_score

def rank_frames(video_frames, method="combined"):
    num_frames = video_frames.shape[2]
    quality_scores = np.zeros(num_frames)

    for i in range(num_frames):
        frame = video_frames[:, :, i]

        norm_frame = frame / frame.max() if frame.max() > 0 else frame

        peak_metric = np.max(norm_frame)
        sharpness_metric = calculate_gradient_sharpness(norm_frame)

        if method == "peak":
            quality_scores[i] = peak_metric

        elif method == "spectral":
            quality_scores[i] = calculate_spectral_quality(norm_frame)

        elif method == "sharpness":
            quality_scores[i] = sharpness_metric

        elif method == "combined_star":
            quality_scores[i] = 0.6 * peak_metric + 0.4 * sharpness_metric #combined sin spectral

        else:   # combined_planet
            # CombinaciÃ³n de espectral y gradiente
            spec = calculate_spectral_quality(norm_frame)
            grad = calculate_gradient_sharpness(norm_frame)
            quality_scores[i] = spec * grad
    # Ordenar descendente
    ranked_indices = np.argsort(quality_scores)[::-1]
    return ranked_indices, quality_scores

def load_fits_frames(folder_path):
    """Carga todos los .fits de un directorio"""
    fits_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".fits", ".fit"))
    ])
    
    fits_files = fits_files[20:]  #caso estrella

    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {folder_path}") #debuggin

    frames = []
    for fname in fits_files:
        path = os.path.join(folder_path, fname)
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                if data.ndim == 2:
                    frames.append(data.astype(np.float32))
                elif data.ndim == 3:
                    for i in range(data.shape[0]):
                        frames.append(data[i, :, :].astype(np.float32))
        except Exception:
            continue
            
    if not frames:
        raise ValueError("No se pudieron cargar datos vÃ¡lidos.") #debuggin
        
    return np.stack(frames, axis=2)

def calculate_gradient_sharpness(frame):
    """
    [Gx, Gy] = gradient(frame);
    sharpness = mean(sqrt(Gx.^2 + Gy.^2));
    """
    gy, gx = np.gradient(frame)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(gradient_magnitude)

def select_best_frames(ranked_indices, percentage):
    N = len(ranked_indices)
    num_select = max(1, int(round(N * percentage / 100)))
    return ranked_indices[:num_select]

def align_frames_standard(frames, reference):
    """
    Alinea los frames respecto a la referencia usando correlaciÃ³n cruzada (FFT).
    """
    num_frames = frames.shape[2]
    aligned = np.zeros_like(frames)
    
    for i in range(num_frames):
        current = frames[:, :, i]
        
        if i == 0 and np.array_equal(current, reference):
            aligned[:, :, i] = current
            continue

        # Registro por correlaciÃ³n cruzada usando FFT
        shift = compute_shift(reference, current)
        
        # Aplicar desplazamiento
        aligned[:, :, i] = shift_image(current, shift)
        
    return aligned

def align_frames_robust(frames, reference, sigma_smooth=0):
    num_frames = frames.shape[2]
    aligned = np.zeros_like(frames)
    
    if sigma_smooth > 0:
        ref_smooth = gaussian_filter(reference, sigma=sigma_smooth)
    else:
        ref_smooth = reference
    
    for i in range(num_frames):
        current = frames[:, :, i]
        
        if i == 0:
            aligned[:, :, i] = current
            continue

        # Suavizar frame actual
        if sigma_smooth > 0:
            current_smooth = gaussian_filter(current, sigma=sigma_smooth)
        else:
            current_smooth = current
            
        # CorrelaciÃ³n usando las versiones suavizadas
        shift = compute_shift(ref_smooth, current_smooth)
        
        # Aplicar el desplazamiento al frame original (no suavizado)
        aligned[:, :, i] = shift_image(current, shift)
        
    return aligned

def compute_shift(ref, img):
    """Calcula el desplazamiento (y, x) usando correlaciÃ³n cruzada FFT."""
    # Cross-correlation en dominio FFT
    f_ref = np.fft.fft2(ref)
    f_img = np.fft.fft2(img)

    # Cross correlation
    cc = np.fft.ifft2(f_ref * f_img.conj()).real
    
    # Encontrar el pico
    shift_y, shift_x = np.unravel_index(np.argmax(cc), cc.shape)
    
    # Ajustar coordenadas para desplazamientos negativos
    if shift_y > ref.shape[0] // 2:
        shift_y -= ref.shape[0]
    if shift_x > ref.shape[1] // 2:
        shift_x -= ref.shape[1]
        
    return np.array([shift_y, shift_x])

def shift_image(img, shift):
    """Desplaza la imagen usando interpolaciÃ³n spline (ndimage)."""
    return ndimage.shift(img, shift, order=3, mode='nearest')

def stack_images(frames, method, weights):
    """Apila las imÃ¡genes alineadas."""
    if method == "mean":
        return np.mean(frames, axis=2)
    
    elif method == "median":
        return np.median(frames, axis=2)
    
    elif method == "weighted":
        # Normalizar pesos
        w = weights / np.sum(weights)
        stacked = np.average(frames, axis=2, weights=w)
        return stacked
    
    else:
        # Fallback a mean
        return np.mean(frames, axis=2)

def apply_safe_deconvolution(image):
    """
    Aplica deconvoluciÃ³n Lucy-Richardson con PSF Gaussiana.
    """
    # Normalizar imagen entre 0 y 1 para estabilidad
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        img_norm = image

    # Crear PSF Gaussiana (sigma=0.8, size=3x3 aprox)
    sigma = 0.8
    size = 3
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf = g / g.sum()

    # DeconvoluciÃ³n 
    deconvolved = richardson_lucy(img_norm, psf, num_iter=5)
    
    # Filtrado suave 
    deconvolved = ndimage.median_filter(deconvolved, size=2)
    
    # Restaurar rango original (aproximado) o dejar en [0,1]
    deconvolved = np.clip(deconvolved, 0, 1)
    
    return deconvolved
