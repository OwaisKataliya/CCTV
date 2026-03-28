"""
kalman.py
---------
Constant-velocity Kalman filter for bounding box tracking.

State vector (8D): [cx, cy, w, h, vx, vy, vw, vh]
Measurement (4D):  [cx, cy, w, h]

The filter predicts the next position of a box using velocity estimated
from previous observations. It corrects the prediction when a new
measurement (detection) is available.
"""

import numpy as np
from tracker.config import KALMAN_PROCESS_NOISE, KALMAN_MEAS_NOISE


class KalmanBox:
    """
    Linear Kalman filter operating on bounding box center + size.
    Assumes constant-velocity motion model.
    """

    def __init__(self, dt: float = 1.0):
        q = KALMAN_PROCESS_NOISE
        r = KALMAN_MEAS_NOISE

        # State transition: position += velocity * dt
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, 4 + i] = dt

        # Process and measurement noise covariances
        self.Q = np.eye(8, dtype=np.float32) * q
        self.R = np.eye(4, dtype=np.float32) * r

        # Measurement matrix: observe [cx, cy, w, h] only (not velocity)
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

    def initiate(self, xywh: np.ndarray):
        """
        Initialise state from the first measurement.

        Returns:
            (x, P) — initial state vector and covariance matrix.
        """
        x      = np.zeros(8, dtype=np.float32)
        x[0:4] = xywh
        P      = np.eye(8, dtype=np.float32) * 10.0
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray):
        """
        Propagate state forward by one time step.

        Returns:
            (x_pred, P_pred)
        """
        x = self.F.dot(x)
        P = self.F.dot(P).dot(self.F.T) + self.Q
        return x, P

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray):
        """
        Correct predicted state with a new measurement z = [cx, cy, w, h].

        Returns:
            (x_corrected, P_corrected)
        """
        S = self.H.dot(P).dot(self.H.T) + self.R
        K = P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(x)
        x = x + K.dot(y)
        P = (np.eye(8, dtype=np.float32) - K.dot(self.H)).dot(P)
        return x, P
