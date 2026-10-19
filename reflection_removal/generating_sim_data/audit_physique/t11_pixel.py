"""Un pixel, tous les termes, a la main."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
NG = 1.52
size, fov, th0, phi = 384, 40.0, np.radians(45), 0.0
h, gap, dbl, d_refl = 6e-3, 0.0, False, 1.0
f = (size/2)/np.tan(np.radians(fov)/2)
nv = np.array([np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0)])
print(f"f_px = {f:.2f}   n_vitre = ({nv[0]:.4f}, {nv[1]:.4f}, {nv[2]:.4f})   "
      f"theta0 = 45 deg   h = 6 mm   d_refl = 1 m\n")

for label, (x, y) in [("centre        ", (0.0, 0.0)),
                      ("bord haut     ", (0.0, -191.5)),
                      ("coin haut-gauche", (-191.5, -191.5))]:
    v = np.array([x, y, f])
    p = v @ nv
    t = v - p*nv
    th = t/np.linalg.norm(t)
    cos_i = abs(p)/np.linalg.norm(v)
    th_i = np.degrees(np.arccos(cos_i))
    tht = np.arcsin(np.sin(np.radians(th_i))/NG)
    walk = 2*h*np.tan(tht)                      # metres
    D = walk*th                                 # vecteur de base 3D, metres
    # --- exact : parallaxe complete
    ox = (f/d_refl)*(D[0] - x*D[2]/f)
    oy = (f/d_refl)*(D[1] - y*D[2]/f)
    # --- terme lateral seul (bonne version)
    lx, ly = (f/d_refl)*D[0], (f/d_refl)*D[1]
    # --- terme radial seul
    rx, ry = -(f/d_refl)*x*D[2]/f, -(f/d_refl)*y*D[2]/f
    # --- formule du code
    nxy = np.hypot(t[0], t[1])
    cx = (f/d_refl)*walk*cos_i*t[0]/nxy
    cy = (f/d_refl)*walk*cos_i*t[1]/nxy
    # --- verification par trace
    ex, ey, _ = RP.ghost_offset_exact(np.array([x]), np.array([y]), f, nv, NG,
                                      h, gap, dbl, d_refl)
    ang = lambda a, b: np.degrees(np.arctan2(b, a))
    print(f"--- {label}  pixel ({x:+.1f}, {y:+.1f}) ---")
    print(f"  theta_i = {th_i:5.2f} deg    marche |D| = {walk*1e3:.4f} mm")
    print(f"  t_hat = ({th[0]:+.4f}, {th[1]:+.4f}, {th[2]:+.4f})   -> D_z = {D[2]*1e3:+.4f} mm")
    print(f"  facteur lateral   : |(t_x,t_y)|/|t| = {nxy/np.linalg.norm(t):.4f}"
          f"   vs  cos_i du code = {cos_i:.4f}")
    print(f"  terme lateral  = ({lx:+8.4f}, {ly:+8.4f})  px")
    print(f"  terme radial   = ({rx:+8.4f}, {ry:+8.4f})  px   <-- ABSENT du code")
    print(f"  somme = exact  = ({ox:+8.4f}, {oy:+8.4f})  px   |.| = {np.hypot(ox,oy):.4f}"
          f"  angle {ang(ox,oy):+7.2f} deg")
    print(f"  trace de Snell = ({ex[0]:+8.4f}, {ey[0]:+8.4f})  px   (controle)")
    print(f"  TON CODE       = ({cx:+8.4f}, {cy:+8.4f})  px   |.| = {np.hypot(cx,cy):.4f}"
          f"  angle {ang(cx,cy):+7.2f} deg")
    print(f"  -> module {np.hypot(cx,cy)/np.hypot(ox,oy)-1:+.1%}, "
          f"direction {ang(cx,cy)-ang(ox,oy):+.2f} deg\n")
