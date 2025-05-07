import math
from config import PIXEL_TO_METER
import numpy as np

def angle_between(p1, p2, p3):
    """Ángulo en p2 formado por (p1–p2) y (p3–p2), en grados."""
    v1 = (p1[0]-p2[0], p1[1]-p2[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1)*math.hypot(*v2)
    return math.degrees(math.acos(max(min(dot/mag,1),-1)))

def pixels_to_meters(pt_px):
    """Convierte un punto (x_px, y_px) a metros: (x_m, y_m)."""
    x_px, y_px = pt_px
    return x_px * PIXEL_TO_METER, y_px * PIXEL_TO_METER

def cartesian_to_polar(pt):
    """De (x, y) cartesianos a (r, θ) polares (θ en rad)."""
    x, y = pt
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta

def interpolate_point(p_elbow, p_wrist, frac):
    """Devuelve el punto a frac·100% del vector codo→muñeca."""
    ex, ey = p_elbow
    wx, wy = p_wrist
    return (ex + frac*(wx - ex), ey + frac*(wy - ey))

def compute_velocities(positions, times):
    """
    positions: list de (x,y)
    times:      list de t
    → list de (vx,vy)
    """
    pos = np.asarray(positions)   # shape (N,2)
    t   = np.asarray(times)       # shape (N,)

    # numpy.gradient puede recibir un array de muestras y los tiempos
    vx = np.gradient(pos[:,0], t)
    vy = np.gradient(pos[:,1], t)

    return list(zip(vx, vy))

def compute_accelerations(velocities, times):
    """
    velocities: list de (vx,vy)
    times:      list de t
    → list de (ax,ay)
    """
    vel = np.asarray(velocities)
    t   = np.asarray(times)

    ax = np.gradient(vel[:,0], t)
    ay = np.gradient(vel[:,1], t)

    return list(zip(ax, ay))

def compute_angular_velocity(thetas, times):
    thetas = np.asarray(thetas)
    t      = np.asarray(times)
    ω = np.gradient(thetas, t)
    return ω.tolist()

def compute_angular_acceleration(omegas, times):
    ω   = np.asarray(omegas)
    t   = np.asarray(times)
    α = np.gradient(ω, t)
    return α.tolist()


def polar_velocity(r, dr_dt, dtheta_dt):
    """
    r: float
    dr_dt: dr/dt
    dtheta_dt: dθ/dt
    → (v_r, v_theta)
    """
    v_r     = dr_dt
    v_theta = r * dtheta_dt
    return v_r, v_theta

def polar_acceleration(r, dr_dt, dtheta_dt, d2r_dt2, d2theta_dt2):
    """
    → (a_r, a_theta)
    """
    a_r      = d2r_dt2 - r * dtheta_dt**2
    a_theta  = r * d2theta_dt2 + 2 * dr_dt * dtheta_dt
    return a_r, a_theta

