#Python 3.11
from lucky_imaging import lucky_imaging, load_fits_frames

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# BLOQUE PRINCIPAL DE EJECUCIÓN
if __name__ == "__main__":
    import os
    # Configuración de rutas
    frames_folder = "/work/alex.aguilera/Light"     # /Light | /Saturno
    output_dir = os.getenv("HOME") + os.path.sep
    output_prefix = "light_result"         #light 
    #output_prefix = "saturno_result"      #sturno
    
    # Parámetros configurables
    params = {
        "selection_method": "sharpness",             # peak | sharpness | spectral | combined_star | combined_planet ###peak no funciona bien para planetas
        "selection_percentage": 5,
        "alignment": True,                           # True | False
        "alignment_type": "standard",                # standard | robust
        "stacking_method": "weighted",               # mean | median | weighted
        "apply_deconvolution": False                 # True | False
    }
        
    # Cargar datos usando la función definida
    frames_vol = load_fits_frames(frames_folder)
    
    # Ejecutar
    result = lucky_imaging(frames_vol, params)

    # Guardar Resultados
    out_base = os.path.join(output_dir, output_prefix)
    method_sel = params["selection_method"]
    method_stack = params["stacking_method"]

    # Guardar FITS
    fits.writeto(out_base + f"_{method_sel}_{method_stack}.fits",result["image"].astype(np.float32),overwrite=True) #importante
    fits.writeto(out_base + "_baseline.fits", result["baseline"].astype(np.float32), overwrite=True)
    np.save(out_base + "_metadata.npy", result["params"])

    #
    best_frame = result["best_frame"]
    lucky_shot = result["image"]
    #plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    #titulo
    fig.suptitle("Alpha Centauri A", fontsize=16, fontweight='bold')

    #contraste
    p_low_b, p_high_b = np.percentile(best_frame, (0.5, 99.5))
    p_low_l, p_high_l = np.percentile(lucky_shot, (0.5, 99.5))
    
    #plot izquierda
    im1 = ax1.imshow(best_frame, cmap='inferno', vmin=p_low_b, vmax=p_high_b, origin='lower')
    ax1.set_title("Mejor Frame Original")
    ax1.axis('off')
    #barra de intensidad izquierda
    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04, label='Intensidad Raw')
    
    #plot derecha
    im2 = ax2.imshow(lucky_shot, cmap='inferno', vmin=p_low_l, vmax=p_high_l, origin='lower')
    ax2.set_title(f"Lucky Shot (Stack {params['selection_percentage']}%)")
    ax2.axis('off')
    #barra de intensidad derecha
    fig.colorbar(im2, ax=ax2, label='Intensidad Normalizada')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    comp_filename = out_base + "_comparacion.png"
    plt.savefig(comp_filename, dpi=150)
