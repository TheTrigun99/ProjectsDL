# genere par extract_notebook.py -- ne pas editer
import sys
sys.path.insert(0, '/home/damien/ProjectsDL/reflection_removal/main_files')
import numpy as np, cv2, random, rawpy, exifread
from pathlib import Path
PATCH, TAU, SEED = 384, 0.1329, 1009
# ===== CELL 27
import matplotlib.pyplot as plt

# --- Color tools  -------------------------------------------------------
D50 = np.array([0.9642, 1.0, 0.8249], np.float32)

BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]], np.float32)
def cat_matrix(src_white, dst_white=D50):
    """Chromatic adaptation of Bradford : XYZ(illuminant src) -> XYZ(dst)."""
    s = BRADFORD @ src_white
    d = BRADFORD @ dst_white
    return (np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD).astype(np.float32)

# Standard Matrix (ICC / IEC 61966-2-1)
XYZ_D50_TO_SRGB = np.array([[3.1338561, -1.6168667, -0.4906146],
                            [-0.9787684, 1.9161415, 0.0334540],
                            [0.0719453, -0.2289914, 1.4052427]], np.float32)
XYZ_D65_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]], np.float32)

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
def apply_matrix(img, M):
    # return np.einsum("ij,...j->...i", M, img)
    return img @ M.T

def luminance(lin_srgb):
    return lin_srgb @ LUM

def srgb_encode(x):
    """Gamma sRGB (linear [0,1] -> encoded [0,1])."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

def srgb_decode(x):
    """Gamma sRGB linear encoded [0,1] -> linear [0,1]
       without a tone curve, it is possible to go to linear sRGB 
    """
    return np.where(x <= 12.92 * 0.0031308, x / 12.92, ((x + 0.055) / 1.055) ** 2.4 )


# --- Display ------------------------------------------------------------
def auto_expose(lin, target=0.18):
    """Expose the mean on the gray."""
    key = float(np.exp(np.log(np.clip(luminance(lin), 1e-6, None)).mean()))
    return lin * (target / max(key, 1e-8))

def to_display(lin_srgb, ev=0.0):
    return srgb_encode(lin_srgb * 2.0 ** ev)

def xyz_to_display(xyz, illum=None, white_balance=True):
    if white_balance:
        w = illum 
        lin = apply_matrix(xyz, XYZ_D50_TO_SRGB @ cat_matrix(w))
    else:
        lin = apply_matrix(xyz, XYZ_D65_TO_SRGB)
    return np.clip(auto_expose(np.clip(lin, 0, None)), 0, 1)

def show_row(images, titles, ev=0.0, size=3.2):
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for ax, img, title in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(to_display(img, ev))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ===== CELL 30
def random_crop(img, size, sat, rng):
    h, w = img.shape[:2]
    C = int(min(h, w) * rng.uniform(0.4, 1.0))
    C = max(C, size)                                  # no upscale
    y = int(rng.integers(0, h - C + 1))
    x = int(rng.integers(0, w - C + 1))
    crop = np.asarray(img[y:y+C, x:x+C], np.float32)    # slicing
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) 
    return out, float(sat[y:y+C, x:x+C].mean())

# ===== CELL 41
GLASS_IOR = 1.52      # refraction value for glass, we have to take this uniform
# SENSOR_W = 0.036      # capteur plein format (m) (format donnant le + de flou) capteur plein format 36x24 mm
# explains why a phone is sharp (small sized sensor; check formula of defocus blur)
SENSOR_full_min = 0.024
SENSOR_W = SENSOR_full_min
SENSOR_APS_C_min  = 0.0157
# We can imagine simuling a few SENSORS (not only aps_c), the main SENSOR of MIT5k is APS-C, 
# but we can simulate other SENSORS
# we take the min cause we have a squared 256x256 / PATCHxPATCH, if we want 3/2 images format we keep SENSOR_W=0.036 m
def _fresnel_single(cos_i, n=GLASS_IOR):
    """Fresnel non polarized, interface air->glass."""
    cos_i = np.clip(cos_i, 1e-6, 1.0)
    sin_t = np.sqrt(np.clip(1 - cos_i ** 2, 0, 1)) / n
    cos_t = np.sqrt(np.clip(1 - sin_t ** 2, 0, 1))
    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t)) ** 2
    return 0.5 * (rs + rp)

def fresnel_angle(theta_i, n=GLASS_IOR):
    """
    Fresnel Formula (paper), same results as _fresnel_single
    """
    theta_i = theta_i * np.pi/180
    theta_ip = np.arcsin(np.sin(theta_i)/n)
    sum = theta_i + theta_ip
    substrat = theta_i - theta_ip
    rs = (np.sin(substrat)/np.sin(sum))**2
    rp = (np.tan(substrat)/np.tan(sum))**2
    return (rs + rp) * 0.5

def fresnel_reflectance(R, n=GLASS_IOR):
    """glass with 2 interfaces (multiple intern reflections) : 2R/(1+R).
       (Geometric sum)
    """
    return 2 * R / (1 + R) 


def sample_scene(rng, size):
    """Draw a camera + a window + distances, derive all of the
    geomtric parameters of the reflection (Sec. B, simplifié : plane window).
    
    explanation:
    theta0: glass angle (pour un pare brise, on a une vitre très inclinée)
    phi: 
    FoV = Field of Vision
    """
    # caméra : FOV 40-90° (mean ~65° Sec. B.2)
    fov = float(rng.uniform(40, 90)) 
    f_px = (size / 2) / np.tan(np.radians(fov) / 2)

    # radius per pixel + normal of the window inclined of theta0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) - (size - 1) / 2
    d = np.stack([xs, ys, np.full_like(xs, f_px)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    # window parameters
    # theta0 = np.radians(float(rng.uniform(0, 55)))

    half_diag = np.arctan(np.sqrt(2) * np.tan(np.radians(fov) / 2))
    theta0_max = np.radians(83.0) - half_diag        # theta_i <= 75 deg partout
    theta0 = float(rng.uniform(0, theta0_max))

    phi = float(rng.uniform(0, 2 * np.pi)) 

    # we move into cartesian coordinates to follow ys, xs
    # by construction nrm  has a borm of 1 https://fr.wikipedia.org/wiki/Coordonn%C3%A9es_sph%C3%A9riques
    nrm = np.array([np.sin(theta0) * np.cos(phi),
                    np.sin(theta0) * np.sin(phi),
                    np.cos(theta0)], np.float32)
    cos_i = (d @ nrm)                      # incidence angle per pixel
    
    # défocus : mise au point sur le sujet transmis, la scène reflétée est
    # virtuellement à d_glass + distance derrière la vitre (miroir)
    f_m = f_px / size * SENSOR_W                 # focale en mètres
    N = float(rng.uniform(1.8, 8.0))             # ouverture
    d_focus = 10.0 ** float(rng.uniform(-0.3, 0.9))          # sujet : 0.5-8 m
    # 0.3 pour ne pas coller la vitre au verre et au plus 3 metres car sinon la vitre n'est pas le
    # principal, mais au max ça vaut d_focus parce que la vitre est entre moi et ce que je photographie
    d_glass = float(rng.uniform(0.3, max(0.4, min(3.0, d_focus))))
    d_refl = d_glass + 10.0 ** float(rng.uniform(0.0, 1.7))  # 1-50 m derrière
    coc = f_m ** 2 / (N * max(d_focus - f_m, 1e-3)) * abs(d_refl - d_focus) / d_refl
    defocus_px = coc / SENSOR_W * size           # cercle de confusion en pixels

    # ghost : décalage latéral de la réflexion sur la 2e face du verre
    h_glass = float(rng.uniform(0.003, 0.015))   # épaisseur du verre (m) (0.003, 0.012) pourquoi 0.0012?

    sin_i = np.sqrt(np.clip((1 - cos_i**2), 0 , None)) # on se limite à [0,pi/2] ? 
    sin_t = sin_i / GLASS_IOR
    tan_t = sin_t / np.sqrt(1 - sin_t ** 2)
    ghost_px = f_px * 2 * h_glass * tan_t * (cos_i**2) / d_glass

    # direction du décalage, PAR PIXEL : projection du rayon sur le plan de la vitre
    tang = d - (d @ nrm)[..., None] * nrm
    tang /= np.maximum(np.linalg.norm(tang, axis=-1, keepdims=True), 1e-8) # on divise pas par 0 ce qui est le cas dans quand la direction
    # des pixels est parallèle à la normale
    ghost_dir = tang[..., :2].astype(np.float32)          # trace dans le plan image
    ghost_dir /= np.maximum(np.linalg.norm(ghost_dir, axis=-1, keepdims=True), 1e-8)
    print(ghost_dir.sum(axis=-1), ghost_dir.shape)
    R1_map =  _fresnel_single(cos_i).astype(np.float32)
    return {"f_px": f_px, "theta0": theta0,
            "direction": (float(np.cos(phi)), float(np.sin(phi))),
            "R_map": fresnel_reflectance(R1_map)[..., None],
            "R1_map": R1_map,
            "defocus_px": float(defocus_px), "ghost_px": (ghost_px.astype(np.float32)), "ghost_dir" : ghost_dir}

def det_params(fov="", theta0="", phi="", N="", d_focus="", d_glass="", d_refl="", h_glass=""):
    """ Choices:
    fov: 40-90 ? 
    theta0: 0-55° ?
    phi: 0-2pi ?
    N: 1.8-8 ?
    d_focus: 0.5-8 m ?
    d_glass: 0.3-3 m ?
    d_refl: 1-50 m ?
    h_glass: 0.003-0.015 m ?
    """
    return {"fov": fov, "theta0": theta0, "phi": phi,
            "aperture": N, "d_focus": d_focus,
            "d_glass": d_glass, "d_refl": d_refl,
            "h_glass": h_glass}


def disk_kernel(radius):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = ((x ** 2 + y ** 2) <= radius ** 2).astype(np.float32)
    return k / k.sum()

def disk_kernel_continuous(radius, ss=8):
    R = int(np.ceil(radius))
    o = (np.arange(ss) + 0.5) / ss - 0.5
    dy, dx = np.meshgrid(o, o, indexing="ij")
    y, x = np.mgrid[-R:R+1, -R:R+1].astype(np.float32)
    k = (((y[..., None, None] + dy) ** 2
          + (x[..., None, None] + dx) ** 2) <= radius ** 2).mean(axis=(2, 3))
    s = k.sum()
    return None if s <= 0 else (k / s).astype(np.float32)


def defocus_continuous(img, diameter):
    """Flou de défocus : noyau disque (bokeh), pas gaussien."""
    radius = diameter/2
    if radius < 0.5:
        return img
    return cv2.filter2D(img, -1, disk_kernel_continuous(radius))

def defocus(img, diameter):
    """Flou de défocus : noyau disque (bokeh), pas gaussien."""
    radius = diameter/2
    if radius < 0.5:
        return img
    return cv2.filter2D(img, -1, disk_kernel(int(np.ceil(radius))))

def double_reflection(img, scene):
    """Fantôme : la 2e réflexion sort décalée d'une quantité qui varie PAR PIXEL —
    nulle là où le rayon est normal à la vitre, croissante ensuite."""
    shift = scene["ghost_px"]                        # carte (h, w)
    if float(shift.max()) < 0.15:                    # tout le champ est négligeable
        return img
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dxy = scene["ghost_dir"]                         # carte (h, w, 2)
    map_x = (xs - dxy[..., 0] * shift).astype(np.float32)
    map_y = (ys - dxy[..., 1] * shift).astype(np.float32)
    ghost = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    a = ((1.0 - scene["R1_map"]) ** ((4 if scene["double_pane"] else 2)))[..., None]    # poids aussi par pixel
    return (img + a * ghost) / (1.0 + a)



# ===== CELL 42
def tilt(rng, lo, hi, g=1.0, toward="lo"):
    """Meme support [lo, hi], densite inclinee vers `toward`.
    g = 1.0  -> uniforme, identique bit a bit a rng.uniform (flux RNG inchange).
    g > 1.0  -> masse deplacee : mediane de u passe de 0.50 a 0.35 (g=1.5),
                0.25 (g=2), 0.17 (g=2.5), 0.12 (g=3)."""
    u = rng.random()
    if g != 1.0:
        u = u ** g if toward == "lo" else 1.0 - (1.0 - u) ** g
    return lo + u * (hi - lo)


# ===== CELL 45
##DEFOCUS <1

BIAS = {"fov": 0.6, "N": 0.6, "d_focus": 0.6,
        "d_glass": 1, "p_double": 0.1,
        "theta0": 1.0, "gap": 1.0, "d_refl": 1.0, "h_glass":1.0}

# ===== CELL 47
GLASS_IOR = 1.52
SENSOR_W = 0.024

_GRID = {}
def _grid(size):
    """(xs, ys, r2) mis en cache : ne dependent que de size."""
    g = _GRID.get(size)
    if g is None:
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) - (size - 1) / 2
        g = _GRID[size] = (np.ascontiguousarray(xs), np.ascontiguousarray(ys),
                           (xs * xs + ys * ys).astype(np.float32))
    return g
            
def sample_scene2(rng, size, biais=False, biais_params=None):
    # --- tirages : ordre identique a l'original ---
    B = biais_params if biais else None
    def T(lo, hi, key, toward="lo"):
        return tilt(rng, lo, hi, B[key] if B else 1.0, toward)

    fov  = float(T(40, 90, "fov"))
    f_px = (size / 2) / np.tan(np.radians(fov) / 2)

    phi  = float(rng.uniform(0, 2 * np.pi))      # AVANT theta0 : theta0_max en depend
    half = np.arctan(np.tan(np.radians(fov)/2) * (abs(np.cos(phi)) + abs(np.sin(phi))))
    theta0_max = np.radians(83.0) - half
    theta0 = float(T(0, theta0_max, "theta0", "hi"))

    N       = float(T(1.4 if biais else 1.8, 8.0, "N"))   # borne basse : seul changement de support
    d_focus = 10.0 ** float(T(-0.3, 0.9, "d_focus"))
    d_hi    = max(0.4, min(3.0, d_focus))
    d_glass = float(np.exp(T(np.log(0.3), np.log(d_hi), "d_glass")))
    d_refl  = d_glass + 10.0 ** float(T(-1.0, 1.7, "d_refl"))
    h_glass = float(T(0.003, 0.020, "h_glass", "hi"))
    gap     = float(T(0.006, 0.020, "gap", "hi"))
    double_pane = rng.random() < (B["p_double"] if B else 0.5)
    
    # --- defocus : purement scalaire ---
    f_m = f_px / size * SENSOR_W
    coc = f_m ** 2 / (N * max(d_focus - f_m, 1e-3)) * abs(d_refl - d_focus) / d_refl
    defocus_px = float(coc / SENSOR_W * size)

    xs, ys, r2 = _grid(size)
    nx = np.float32(np.sin(theta0) * np.cos(phi))
    ny = np.float32(np.sin(theta0) * np.sin(phi))
    nz = np.float32(np.cos(theta0))

    # p = v.n avec v = (xs, ys, f_px) NON normalise
    p = xs * nx
    p += ys * ny
    p += np.float32(f_px * nz)

    # inv = 1/|v| : evite de materialiser d (H,W,3) et le norm sur axis=-1
    inv = r2 + np.float32(f_px * f_px)
    np.sqrt(inv, out=inv)
    np.reciprocal(inv, out=inv)

    cos_i = np.abs(p, dtype=np.float32) * inv
    np.clip(cos_i, 1e-6, 1.0, out=cos_i)

    # --- ghost_dir : le facteur inv se simplifie a la normalisation 2D ---
    tx = xs - p * nx
    ty = ys - p * ny
    nrm2 = tx * tx
    nrm2 += ty * ty
    np.sqrt(nrm2, out=nrm2)
    np.maximum(nrm2, 1e-8, out=nrm2)
    tx /= nrm2
    ty /= nrm2
    ghost_dir = np.stack([tx, ty], -1)

    # --- Fresnel + tan_t partagent sin_t et cos_t ---
    sin_t = cos_i * cos_i
    np.subtract(np.float32(1.0), sin_t, out=sin_t)
    np.sqrt(sin_t, out=sin_t)
    sin_t /= np.float32(GLASS_IOR)
    cos_t = sin_t * sin_t
    np.subtract(np.float32(1.0), cos_t, out=cos_t)
    np.sqrt(cos_t, out=cos_t)

    ncos_t = cos_t * np.float32(GLASS_IOR)
    rs = (cos_i - ncos_t) / (cos_i + ncos_t)
    rs *= rs
    ncos_i = cos_i * np.float32(GLASS_IOR)
    rp = (ncos_i - cos_t) / (ncos_i + cos_t)
    rp *= rp
    rs += rp
    R1_map = rs
    R1_map *= np.float32(0.5)


    R_map = R1_map * np.float32(2.0)
    R_map /= (R1_map + np.float32(1.0))
    tan_t = sin_t/cos_t
    path = h_glass * tan_t                      
    if double_pane:
        tan_i = np.sqrt(np.maximum(1 - (cos_i**2), 1e-8)) / np.maximum(cos_i, 1e-8)
        path = path + gap * tan_i                 # air : angle PLEIN, pas de réfraction
    ghost_px = (f_px * 2 * path * (cos_i**2) / d_refl).astype(np.float32) # modified from d_glass->d_refl so less ghosting

    return {"f_px": f_px, "theta0": theta0,
            "direction": (float(np.cos(phi)), float(np.sin(phi))),
            "R_map": R_map[..., None], "R1_map": R1_map,
            "defocus_px": defocus_px,
            "ghost_px": ghost_px, "ghost_dir": ghost_dir, "double_pane" : double_pane}

def deterministic_scene2(rng, size, params):
    # --- draws ---
    if params["fov"]:
        fov = params["fov"]
    else:
        fov = float(rng.uniform(40, 90))

    f_px = (size / 2) / np.tan(np.radians(fov) / 2)
    if params["phi"]:
        phi = params["phi"]
    else:
        phi = float(rng.uniform(0, 2 * np.pi))
    half = np.arctan(np.tan(np.radians(fov)/2) * (abs(np.cos(phi)) + abs(np.sin(phi))))
    theta0_max = np.radians(83.0) - half     
    if params["theta0"]:
        theta0 = params["theta0"]
        assert theta0<= theta0_max, "theta0 > theta0_max"
    else:    
        theta0 = float(rng.uniform(0, theta0_max))
    if params["N"]:
        N = params["N"]
    else:
        N = float(rng.uniform(1.8, 8.0))
    if params["d_focus"]:
        d_focus = params["d_focus"]
    else:
        d_focus = 10.0 ** float(rng.uniform(-0.3, 0.9))
    d_hi    = max(0.4, min(3.0, d_focus))
    if params["d_glass"]:
        d_glass = params["d_glass"]
    else:
        d_glass = float(np.exp(rng.uniform(np.log(0.3), np.log(d_hi))))
    if params["d_refl"]:
        d_refl = d_glass + params["d_refl"]
    else:
        d_refl = d_glass + 10.0 ** float(rng.uniform(-1.0, 1.7))
    if params["h_glass"]:
        h_glass = params["h_glass"]
    else:
        h_glass = float(rng.uniform(0.008, 0.020))

    # --- defocus ---
    f_m = f_px / size * SENSOR_W
    coc = f_m ** 2 / (N * max(d_focus - f_m, 1e-3)) * abs(d_refl - d_focus) / d_refl
    defocus_px = float(coc / SENSOR_W * size)

    xs, ys, r2 = _grid(size)
    nx = np.float32(np.sin(theta0) * np.cos(phi))
    ny = np.float32(np.sin(theta0) * np.sin(phi))
    nz = np.float32(np.cos(theta0))

    p = xs * nx
    p += ys * ny
    p += np.float32(f_px * nz)

    
    inv = r2 + np.float32(f_px * f_px)
    np.sqrt(inv, out=inv)
    np.reciprocal(inv, out=inv)

    cos_i = np.abs(p, dtype=np.float32) * inv
    np.clip(cos_i, 1e-6, 1.0, out=cos_i)

    # --- ghost_dir ---
    tx = xs - p * nx
    ty = ys - p * ny
    nrm2 = tx * tx
    nrm2 += ty * ty
    np.sqrt(nrm2, out=nrm2)
    np.maximum(nrm2, 1e-8, out=nrm2)
    tx /= nrm2
    ty /= nrm2
    ghost_dir = np.stack([tx, ty], -1)

    # --- Fresnel ---
    sin_t = cos_i * cos_i
    np.subtract(np.float32(1.0), sin_t, out=sin_t)
    np.sqrt(sin_t, out=sin_t)
    sin_t /= np.float32(GLASS_IOR)
    cos_t = sin_t * sin_t
    np.subtract(np.float32(1.0), cos_t, out=cos_t)
    np.sqrt(cos_t, out=cos_t)

    ncos_t = cos_t * np.float32(GLASS_IOR)
    rs = (cos_i - ncos_t) / (cos_i + ncos_t)
    rs *= rs
    ncos_i = cos_i * np.float32(GLASS_IOR)
    rp = (ncos_i - cos_t) / (ncos_i + cos_t)
    rp *= rp
    rs += rp
    R1_map = rs
    R1_map *= np.float32(0.5)

    R_map = R1_map * np.float32(2.0)
    R_map /= (R1_map + np.float32(1.0))
          
    tan_t = sin_t/cos_t
    path = h_glass * tan_t                      

    dp = params.get("double_pane")
    double_pane = (rng.random() < 0.5) if dp is None else dp
    if double_pane:
        gap = params["gap"] if params.get("gap") else float(rng.uniform(0.006, 0.020))
        gap = float(rng.uniform(0.006, 0.020))  # air gap
        tan_i = np.sqrt(np.maximum(1 - (cos_i**2), 1e-8)) / np.maximum(cos_i, 1e-8)
        path = path + gap * tan_i              
    ghost_px = (f_px * 2 * path * (cos_i**2) / d_refl).astype(np.float32) #modified d_glass -> d_refl

    return {"f_px": f_px, "theta0": theta0,
            "direction": (float(np.cos(phi)), float(np.sin(phi))),
            "R_map": R_map[..., None], "R1_map": R1_map,
            "defocus_px": defocus_px,
            "ghost_px": ghost_px, "ghost_dir": ghost_dir, "double_pane" : double_pane}    

def sample_scene_lowres(rng, size, coarse=128):
    s = sample_scene2(rng, coarse)
    k = size / coarse                       # f_px scales with resolution: rescale both
    up = lambda a: cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
    return {"R_map": up(s["R_map"][..., 0])[..., None], "R1_map": up(s["R1_map"]),
            "defocus_px": s["defocus_px"] * k,      # <- pixel quantities, must scale
            "ghost_px":   up(s["ghost_px"]) * k,    # <- idem
            "ghost_dir":  up(s["ghost_dir"]), "double_pane": s["double_pane"]}


# ===== CELL 50
SAT_TOL = 0.02
def compute_exposure(m, t, r, sat_t, sat_r, M, e_lock=None):
    """
    S1 function
    We're USING TAU = srgb_decode((0.4)) = 0.1329
    have to delete context
    """
    m_s, t_s, r_s = apply_matrix(m, M), apply_matrix(t, M), apply_matrix(r, M)
    if e_lock is not None:                 
        e = e_lock
    elif sat_t <= SAT_TOL and sat_r <= SAT_TOL:
        e = TAU / max(m_s.mean(), 1e-8)
    else:
        e = 1.0 / (min(float(t_s.max()), float(r_s.max())))
    return m_s * e, t_s * e, r_s * e, e  

# ===== CELL 51
def apply_exposure(t, M, e):
    return apply_matrix(t, M) * e

def simulate_example(src_t, src_r, rng, patch=None, tau=None, params={}, deterministic=False):
    """Func. 1 du papier. src_t, src_r = entrées du pool (xyz, e, illum...).
    Retourne (m, t, r, c) en sRGB linéaire."""
    patch = patch or PATCH
    tau = tau or TAU

    # 1-2) découpes : t depuis l'image i ; (r, c) = moitiés disjointes de j
    sat_t, sat_r = src_t["saturation"], src_r["saturation"]
    t, frac_t = random_crop(src_t["xyz"], patch, sat_t, rng)

    r, frac_r= random_crop(src_r["xyz"], patch, sat_r, rng)
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])


    # 3) dé-exposition : pixels proportionnels à la luminance de scène
    t = t / src_t["e"]
    if src_t["ibl"]:
        pass 
    else: r = r / src_r["e"]

    true_t = t.copy()
    # 4) simulation géométrique : caméra + vitre échantillonnées (Sec. B)
    if deterministic:
        assert params, "deterministic=True requires params"
        scene = deterministic_scene2(rng, patch, params)
    else:
        scene = sample_scene2(rng, patch)
    # on retourne l'image pour l'effet miroir quand on prend une photo
    if rng.random() < 0.5:
        r = r[:, ::-1].copy()                    # miroir (c reste une vue directe)
    # if scene["defocus_px"]<1 and rng.random()<0.5:
    #     defo = np.ceil(scene["defocus_px"])
    # else:
    defo = scene["defocus_px"]
    r = defocus_continuous(r, defo)
    r = double_reflection(r, scene)       # fantôme des deux faces
    r = r * scene["R_map"]                       # Fresnel par pixel
    t = t * (1.0 - scene["R_map"])               # la vitre atténue aussi t
    # 5) additive mixture
    m = t + r # it is like alpha * R + (1-alpha)*T 
    # 6) ré-exposition (Func. S1)
    m, t, r, expo= compute_exposure(m, t, r, frac_t, frac_r, M)
    m, t, r, true_t = (np.clip(x, 0.0, 1.0) for x in (m, t, r, apply_exposure(t=true_t,M=M, e=expo)))

    params = {}
    params["f_px"] = scene["f_px"]
    params["theta0"] = scene["theta0"]
    params["direction"] = scene["direction"]
    params["ghost_px"] = scene["ghost_px"].mean()
    params["defocus_px"] = scene["defocus_px"]
    params["double_pane"] = scene["double_pane"]
    params["scene_map"] = scene["R_map"]
    params["exposure"] = expo

    return {"m": m, "t": t, "r": r, "params" : params, "true_t": true_t}

# ===== CELL 52
def make_t_crop(src_t, rng, patch=None):
    """Le crop transmis, isole du reste : c'est lui qu'on partage dans un groupe."""
    patch = patch or PATCH
    return random_crop(src_t["xyz"], patch, src_t["saturation"], rng)


def simulate_example_k(src_t, src_r, rng, patch=None, tau=None, params={},
                     deterministic=False, t_pre=None, e_lock=None, scene=None):
    """Func. 1 du papier. src_t, src_r = entrées du pool (xyz, e, illum...).
    Retourne (m, t, r, c) en sRGB linéaire."""
    patch = patch or PATCH
    tau = tau or TAU

    sat_t, sat_r = src_t["saturation"], src_r["saturation"]
    if t_pre is None:
        t, frac_t = random_crop(src_t["xyz"], patch, sat_t, rng)
    else:
        t, frac_t = t_pre

    r, frac_r = random_crop(src_r["xyz"], patch, sat_r, rng)
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])

    t, r = t / src_t["e"], r / src_r["e"]
    true_t = t.copy()
    if scene is None:
        if deterministic:
            assert params, "deterministic=True requires params"
            scene = deterministic_scene2(rng, patch, params)
        else:
            scene = sample_scene2(rng, patch, biais=True, biais_params=BIAS)
        # # le pile-ou-face du defocus fait partie de la vitre : il se fige avec elle
        # scene["defo"] = (np.ceil(scene["defocus_px"])
        #                  if scene["defocus_px"] < 1 and rng.random() < 0.5
        #                  else scene["defocus_px"])
        
    if rng.random() < 0.5:
        r = r[:, ::-1].copy()                   
    r = defocus_continuous(r, scene["defocus_px"])
    r = double_reflection(r, scene)       
    r = r * scene["R_map"]                       
    t = t * (1.0 - scene["R_map"])               
    # 5) additive mixture
    m = t + r
    # 6) ré-exposition (Func. S1)
    m, t, r, expo = compute_exposure(m, t, r, frac_t, frac_r, M, e_lock)
    m, t, r, true_t = (np.clip(x, 0.0, 1.0) for x in (m, t, r, apply_exposure(t=true_t, M=M, e=expo)))   

    params = {"f_px": scene["f_px"], "theta0": scene["theta0"],
              "direction": scene["direction"],
              "ghost_px": float(scene["ghost_px"].mean()),
              "defocus_px": scene["defocus_px"],
              "double_pane": scene["double_pane"],
              "exposure": expo}

    return {"m": m, "t": t, "r": r, "params": params, "true_t": true_t}


# ===== CELL 53
def simulate_example_enc(src_t, src_r, rng, patch=None, tau=None, params={},
                     deterministic=False, t_pre=None, r_pre=None, e_lock=None, scene=None, bias_dict=BIAS):
    patch = patch or PATCH
    tau = tau or TAU

    sat_t, sat_r = src_t["saturation"], src_r["saturation"]
    if t_pre is None:
        t, frac_t = random_crop(src_t["xyz"], patch, sat_t, rng)
    else:
        t, frac_t = t_pre
    if r_pre is None:
        r, frac_r = random_crop(src_r["xyz"], patch, sat_r, rng)
        if rng.random() < 0.5:
            r = r[:, ::-1].copy() 
    else:
        r, frac_r = r_pre
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])

    r = r / src_r["e"]
    t = t / src_t["e"]
    true_t = t.copy()
    if scene is None:
        if deterministic:
            assert params, "deterministic=True requires params"
            scene = deterministic_scene2(rng, patch, params)

        else:
            scene = sample_scene2(rng, patch, biais=True, biais_params=bias_dict)
                          
    r = defocus_continuous(r, scene["defocus_px"])
    r = double_reflection(r, scene)       
    r = r * scene["R_map"]                       
    t = t * (1.0 - scene["R_map"])               
    # 5) additive mixture
    m = t + r
    # 6) ré-exposition (Func. S1)
    m, t, r, expo = compute_exposure(m, t, r, frac_t, frac_r, M, e_lock)
    m, t, r, true_t = (np.clip(x, 0.0, 1.0) for x in (m, t, r, apply_exposure(t=true_t, M=M, e=expo)))   

    params = {"f_px": scene["f_px"], "theta0": scene["theta0"],
              "ghost_px": (scene["ghost_px"].astype(np.float32)),
              "defocus_px": scene["defocus_px"],
              "double_pane": scene["double_pane"],
              "exposure": expo, "R_map" : scene["R_map"]}

    return {"m": m, "t":t, "r": r, "params": params, "true_t": true_t}


# ===== CELL 55
def ssim_fast(m, t, data_range=1.0):
    """skimage default (uniform 7x7 + sample covariance), 4.5x faster."""
    C1, C2 = 1e-4, 9e-4                             
    cn = 49 / 48                                     
    f = lambda im: cv2.boxFilter(im, -1, (7, 7), normalize=True,
                                 borderType=cv2.BORDER_REFLECT)
    ux, uy = f(m), f(t)
    vx  = cn * (f(m * m) - ux * ux)
    vy  = cn * (f(t * t) - uy * uy)
    vxy = cn * (f(m * t) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))

    w = np.array([m[..., k].mean() for k in range(3)], np.float64)  
    w /= max(w.sum(), 1e-8)
    S = (S[3:-3, 3:-3] * w).sum(axis=-1)           
    return float(S.mean()), float(S.std())

# ===== CELL 57
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1 / max(mse, 1e-10))

# ===== CELL 58
CULL_SIZE = 256

# ===== CELL 59
from acr_tone_fast import finish
from dataset_sim import square_resize, jpeg_roundtrip
u8 =lambda a: square_resize(finish(a, ev=0.0, tone=True), CULL_SIZE, None)
transform = lambda b: jpeg_roundtrip(u8(b), 95, "444").astype(np.float32) / 255.

def psnr_raw_to_jpeg(M,T):
    M, T = transform(M), transform(T)
    return 10 * np.log10(1. / max(float(((M - T) ** 2).mean()), 1e-12))

# ===== CELL 61
def lum(x):
    return np.ascontiguousarray(x.astype(np.float32) @ LUM)

S, P, SIG = 256, 32, 8.0
def hf(Y, sig=SIG):
    return Y - cv2.GaussianBlur(Y, (0, 0), sig)


def cases(X, f):
    n = S // P
    return f(X.reshape(n, P, n, P).transpose(0, 2, 1, 3).reshape(n, n, P * P), axis=-1)


def rs(x):
    return cv2.resize(np.ascontiguousarray(x.astype(np.float32)), (S, S), interpolation=cv2.INTER_AREA)


def descripteurs(m, tp, r):
    """m, tp (= transmission attenuee, dans m), r (= reflet dans m) : lineaires HxWx3, m ~ tp + r."""
    Ym, Yt, Yr = lum(rs(m)), lum(rs(tp)), lum(rs(r))
    Ht, Hr = hf(Yt), hf(Yr)
    st, sr = cases(Ht, np.std), cases(Hr, np.std)
    mu = cases(Ym, np.mean)
    ct, cr = st / (mu + 1e-3), sr / (mu + 1e-3)            # contrastes de Weber locaux
    rho = sr / (st + 1e-6)
    lrho = np.log2(np.clip(rho, 1 / 64, 64))
    part_r = cases(Yr, np.mean) / (mu + 1e-6)              # part du reflet dans la luminance de m
    actif = (ct + cr) > 0.02                               # cases ou quelque chose se voit
    d = dict(
        ratio=float(Hr.std() / (Ht.std() + 1e-8)),         # = ratio du notebook
        A_rdom=float((rho > 1).mean()),                    # aire ou le reflet domine la structure
        A_rdom2=float((rho > 2).mean()),
        A_rdom4=float((rho > 4).mean()),                   # transmission masquee (quasi invisible)
        A_tdom=float((rho < 1).mean()),                    # aire ou la transmission domine
        A_tdom2=float((rho < 0.5).mean()),                 # transmission nettement dominante (ancre)
        A_tvis=float((ct > 0.03).mean()),                  # transmission visible en absolu (Weber > 3 %)
        A_tvis_rel=float(((ct > 0.03) & (rho < 2)).mean()),  # visible ET pas ecrasee par le reflet
        rho_med=float(np.median(rho)), lrho_mean=float(lrho.mean()),
        lrho_p10=float(np.percentile(lrho, 10)), lrho_p90=float(np.percentile(lrho, 90)),
        lrho_std=float(lrho.std()),                        # heterogeneite spatiale de la domination
        part_r=float(Yr.mean() / (Ym.mean() + 1e-6)),      # part globale du reflet dans la luminance
        A_rbrillant=float((part_r > 0.5).mean()),          # aire ou le reflet est plus lumineux que t'
        ct_med=float(np.median(ct)), cr_med=float(np.median(cr)),
        t_std=float(Yt.std()), t_hf=float(Ht.std()), r_hf=float(Hr.std()),
        A_actif=float(actif.mean()),
    )
    return d

# ===== CELL 62
CULL_SIZE = 256 # 384
vis_min, vis_max = 0.01, 0.6 # vis baseline = 0.08 # apr_s j'ai 0.02, 0.5

def measure(exa):
    """Toutes les quantités de tri en une passe, plus le premier critère qui échoue.
    Remplace is_useful : is_useful(ex) <=> measure(ex)["verdict"] == "ok"."""

    rs = lambda x: cv2.resize(x, (CULL_SIZE, CULL_SIZE), interpolation=cv2.INTER_AREA)
    mc, tc, rc = rs(exa["m"]), rs(exa["t"]), rs(exa["r"])
    Ym, Yt, Yr = luminance(mc), luminance(tc), luminance(rc)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)

    Yr_hf = Yr - cv2.GaussianBlur(Yr, (0, 0), 8)
    Yt_hf = Yt - cv2.GaussianBlur(Yt, (0, 0), 8)
    ratio = float(Yr_hf.std() / (Yt_hf.std() + 1e-8))
    defocus_px = exa["params"]["defocus_px"]
                  
    d = {"ssim": ssim_m_t, "ssim_std": std_m_t, "Ym": float(Ym.mean()),
          "m_mean": float(mc.mean()),
         "t_std": float(Yt.std()), "r_std": float(Yr.std()),
         "r_hf": float(Yr_hf.std()), "ratio": ratio,
         "sat_frac": float((mc.max(axis=-1) > 0.99).mean())} # , "jpeg_psnr": float(psnr_raw_to_jpeg(mc,tc))}
    if   not 0.40 < d["ssim"] < 0.96:        d["verdict"] = "ssim"
    elif not 0.05 <= d["ssim_std"] <= 0.30:  d["verdict"] = "ssim_std"
    elif d["r_std"] < 0.005:                 d["verdict"] = "voile"
    elif ratio > 2 and defocus_px < 1.0:     d["verdict"] = "reflet_dominant"
    elif not 0.04 < d["Ym"] < 0.20:          d["verdict"] = "exposition"
    elif d["m_mean"] >= 0.7:                 d["verdict"] = "m_mean"
    elif d["t_std"] < 0.015:                 d["verdict"] = "t_plat" 
    else:                                    d["verdict"] = "ok"
    return d

def measure_a_tdom(exa):
    rs = lambda x: cv2.resize(x, (CULL_SIZE, CULL_SIZE), interpolation=cv2.INTER_AREA)
    mc, tc, rc = rs(exa["m"]), rs(exa["t"]), rs(exa["r"])
    Ym, Yt, Yr = luminance(mc), luminance(tc), luminance(rc)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)

    Yr_hf = Yr - cv2.GaussianBlur(Yr, (0, 0), 8)
    Yt_hf = Yt - cv2.GaussianBlur(Yt, (0, 0), 8)
    st, sr = cases(Yt_hf, np.std), cases(Yr_hf, np.std)
    rho = sr / (st + 1e-6)    
    d = dict(                    # aire ou le reflet domine la structure
    A_rdom2=float((rho > 2).mean()),
    A_rdom4=float((rho > 4).mean()),                   # transmission masquee (quasi invisible)
    A_tdom=float((rho < 1).mean()),                    # aire ou la transmission domine
    A_tdom2=float((rho < 0.5).mean()),
    part_r=float(Yr.mean() / (Ym.mean() + 1e-6)))
    mt, mr = cases(Yt, np.mean), cases(Yr, np.mean)   
    part_loc = mr / (mr + mt)             
    p90_part_loc = np.percentile(part_loc, 90)
    ratio_lum = float(Yr.mean()/Yt.mean())    
    if not (0.4<d["A_tdom"]<0.65):
        d["verdict"] = "A_tdom "
        return d 
    elif ratio_lum >1:
        d["verdict"]= "ratio_lum"
        return d
    elif p90_part_loc >0.8:
        d["verdict"]  = "p90_part_loc"
        return d
    elif d["part_r"]>0.50:
        d["verdict"]= "part_r"
        return d
    
    elif  d["A_rdom2"]<0.20:
        d["verdict"] = "A_rdom2"
        return d
    
    elif  0.5>ssim_m_t:
        d["verdict"] = "ssim"
        return d    
    elif (float(Yr.std()))<0.07:
        d["verdict"] = "voile_uniforme"
        return d
    
    elif not 0.05<std_m_t <0.35:
        d["verdict"] = "ssim_std"
        return d
    
    elif not 0.04 < float(Ym.mean()) < 0.3:
        d["verdict"] = "exposition"
        return d
    
    elif float(mc.mean()) > 0.7:
        d["verdict"] = "mean"
        return d
    
    elif float(Yt.std()) < 0.020:
        d["verdict"] = "t_std"
        return d
    
    else:

        d["verdict"]="ok"
        return d
    
def is_useful(exa):
    return measure(exa)["verdict"] == "ok"