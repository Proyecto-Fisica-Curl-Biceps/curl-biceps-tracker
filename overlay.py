import cv2
import matplotlib.pyplot as plt

class Overlay:
    @staticmethod
    def draw_arm_skeleton(frame, landmarks):
        """
        Dibuja el esqueleto del brazo (hombro, codo y muñeca) sobre el frame.
        """
        shoulder = landmarks.get('RIGHT_SHOULDER')
        elbow = landmarks.get('RIGHT_ELBOW')
        wrist = landmarks.get('RIGHT_WRIST')

        for pt in (shoulder, elbow, wrist):
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

        if shoulder and elbow:
            cv2.line(frame,
                     (int(shoulder[0]), int(shoulder[1])),
                     (int(elbow[0]), int(elbow[1])),
                     (255, 0, 0), 2)
        if elbow and wrist:
            cv2.line(frame,
                     (int(elbow[0]), int(elbow[1])),
                     (int(wrist[0]), int(wrist[1])),
                     (255, 0, 0), 2)
        return frame

    @staticmethod
    def draw_angle(frame, landmarks, angle):
        """
        Pinta el ángulo calculado en el codo encima del frame.
        """
        elbow = landmarks.get('RIGHT_ELBOW')
        if elbow:
            x, y = int(elbow[0]), int(elbow[1])
            cv2.putText(
                frame,
                f"Codo(rad): {angle:.1f}",
                (x - 50, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        return frame

    @staticmethod
    def plot_polar(r_values, theta_values, show=True, save_path=None):
        """
        Genera una gráfica polar de r vs θ.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(theta_values, r_values)
        ax.set_title('Trayectoria polar de la muñeca')

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)
