import os
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from video_processor import iter_frames
from pose_extractor import PoseExtractor
from kinematics import (
    interpolate_point, cartesian_to_polar,
    compute_velocities, compute_accelerations,
    compute_angular_velocity, compute_angular_acceleration,
    angle_between
)
from overlay import Overlay
from config import COM_FRACTION, PIXEL_TO_METER, ARROW_SIZE, VIDEO_PATH

def main():
    os.makedirs("outputs", exist_ok=True)

    # sincronizar reproducción
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps<= 0
        fps= 30.0
    delay = int(1000 / fps)

    ext = PoseExtractor()
    raw_wrist, raw_elbow, times = [], [], []

    prev_time = time.time()
    for t, frame in iter_frames():
        lm = ext.extract(frame)
        if not lm:
            continue

        raw_wrist.append(lm['RIGHT_WRIST'])
        raw_elbow.append(lm['RIGHT_ELBOW'])
        times.append(t)

        # cálculo velo/acc en px
        scale_v= 1.15
        scale_a = 5 
        vel_px = (ARROW_SIZE, ARROW_SIZE)
        acc_px = (ARROW_SIZE, ARROW_SIZE)
        if len(raw_wrist) >= 2:
            dx = raw_wrist[-1][0] - raw_wrist[-2][0]
            dy = raw_wrist[-1][1] - raw_wrist[-2][1]
            vel_px = (dx*scale_v, dy*scale_v)
        if len(raw_wrist) >= 3:
            dx1 = raw_wrist[-2][0] - raw_wrist[-3][0]
            dy1 = raw_wrist[-2][1] - raw_wrist[-3][1]
            acc_px = ((vel_px[0] - dx1)*scale_a,( vel_px[1] - dy1)*scale_a)

        # ángulo del codo
        ang = angle_between(
            lm['RIGHT_SHOULDER'], lm['RIGHT_ELBOW'], lm['RIGHT_WRIST']
        )

        # overlay
        frame_overlay = frame.copy()
        frame_overlay = Overlay.draw_arm_skeleton(frame_overlay, lm)
        frame_overlay = Overlay.draw_angle(frame_overlay, lm, ang)
        start = (int(lm['RIGHT_WRIST'][0]), int(lm['RIGHT_WRIST'][1]))
        end_v = (start[0] + int(vel_px[0]), start[1] + int(vel_px[1]))
        end_a = (start[0] + int(acc_px[0]), start[1] + int(acc_px[1]))
        cv2.arrowedLine(frame_overlay, start, end_v, (0,0,255), 2, tipLength=0.3)
        cv2.arrowedLine(frame_overlay, start, end_a, (255,0,255), 2, tipLength=0.3)

        # FPS real
        curr_time = time.time()
        actual_fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame_overlay, f"FPS: {actual_fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow('Curl Tracker', frame_overlay)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # 1) píxeles → metros
    e_s = [((x - p[0]) * PIXEL_TO_METER,
            (y - p[1]) * PIXEL_TO_METER)
           for (x, y), p in zip(raw_wrist, raw_elbow)]

    # 2) suavizado Savitzky–Golay
    xs, ys = zip(*e_s)
    xs = np.array(xs); ys = np.array(ys)
    win = min(7, len(xs) if len(xs) % 2 else len(xs)-1)
    xs_s = savgol_filter(xs, win, 2)
    ys_s = savgol_filter(ys, win, 2)
    e_s_s = list(zip(xs_s, ys_s))

    # 3) derivadas
    v_s = compute_velocities(e_s_s, times)
    a_s = compute_accelerations(v_s, times)

    # 4) recorte de bordes (suavizado)
    idx0, idx1 = 2, len(times) - 2
    t_cut = times[idx0:idx1]
    xs_cut = xs_s[idx0:idx1]
    ys_cut = ys_s[idx0:idx1]
    vxs_cut, vys_cut = zip(*v_s[idx0:idx1])
    axs_cut, ays_cut = zip(*a_s[idx0:idx1])

    # --- Cartesianas suavizadas ---
    plt.figure(); plt.plot(t_cut, xs_cut, label='x_s(t)'); plt.plot(t_cut, ys_cut, label='y_s(t)')
    plt.title('Posición Cartesianas (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/posicion_cartesianas_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, vxs_cut, label='vx_s(t)'); plt.plot(t_cut, vys_cut, label='vy_s(t)')
    plt.title('Velocidad Cartesianas (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/velocidad_cartesianas_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, axs_cut, label='ax_s(t)'); plt.plot(t_cut, ays_cut, label='ay_s(t)')
    plt.title('Aceleración Cartesianas (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/aceleracion_cartesianas_suav.png", dpi=300); plt.close()

    # --- Polares ---
    # posiciones polares
    r_list, theta_list = zip(*[cartesian_to_polar(pt) for pt in e_s_s])
    dr_dt_list    = compute_angular_velocity(r_list,    times)
    d2r_dt2_list  = compute_angular_acceleration(dr_dt_list, times)
    dtheta_dt_list  = compute_angular_velocity(theta_list, times)
    d2theta_dt2_list = compute_angular_acceleration(dtheta_dt_list, times)

    # recorte
    r_cut      = r_list[idx0:idx1]
    theta_cut  = theta_list[idx0:idx1]
    dr_cut     = dr_dt_list[idx0:idx1]
    d2r_cut    = d2r_dt2_list[idx0:idx1]
    dtheta_cut = dtheta_dt_list[idx0:idx1]
    d2theta_cut= d2theta_dt2_list[idx0:idx1]

    # gráficos polares
    plt.figure(); plt.plot(t_cut, r_cut, label='r_s(t)'); plt.plot(t_cut, theta_cut, label='θ_s(t)')
    plt.title('Posición Polares (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/posicion_polares_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, dr_cut, label='ṙ_s(t)'); plt.plot(t_cut, dtheta_cut, label='ω_s(t)')
    plt.title('Velocidad Polares (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/velocidad_polares_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, d2r_cut, label='¨r_s(t)'); plt.plot(t_cut, d2theta_cut, label='α_s(t)')
    plt.title('Aceleración Polares (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/aceleracion_polares_suav.png", dpi=300); plt.close()

    # --- Dinámica: tangencial, centrípeta ---
    m = 5.0  # masa de la mancuerna en kg
    v_t = [r*ω for r,ω in zip(r_list, dtheta_dt_list)]
    a_t = [r*α + 2*dr*ω for r,dr,ω,α in zip(r_list, dr_dt_list, dtheta_dt_list, d2theta_dt2_list)]
    a_c = [r*(ω**2) for r,ω in zip(r_list, dtheta_dt_list)]


    # recortar
    v_t_cut   = v_t[idx0:idx1]
    a_t_cut   = a_t[idx0:idx1]
    a_c_cut   = a_c[idx0:idx1]


    # gráficos dinámicos
    plt.figure(); plt.plot(t_cut, v_t_cut, label='v_t (m/s)')
    plt.title('Velocidad Tangencial (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/velocidad_tangencial_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, a_t_cut, label='a_t (m/s²)')
    plt.title('Aceleración Tangencial (suavizada)'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/aceleracion_tangencial_suav.png", dpi=300); plt.close()

    plt.figure(); plt.plot(t_cut, a_c_cut, label='a_c (m/s²)')
    plt.title('Aceleración Centrípeta'); plt.xlabel('Tiempo (s)'); plt.legend()
    plt.savefig("outputs/aceleracion_centripeta.png", dpi=300); plt.close()



if __name__ == '__main__':
    main()
