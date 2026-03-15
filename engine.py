import numpy as np
from scipy.integrate import cumulative_trapezoid

class VibrationModel:
    def __init__(self, machine_group='Group1_Rigid'):
        # กำหนดเกณฑ์ตามตาราง ISO 10816-3
        self.thresholds = {
            'Group1_Rigid': [2.3, 4.5, 7.1],
            'Group2_Rigid': [1.4, 2.8, 4.5]
        }
        self.limits = self.thresholds.get(machine_group)

    def process_data(self, time_ms, accel_g):
        # 1. Unit Conversion
        t_sec = np.array(time_ms) / 1000.0
        a_ms2 = (np.array(accel_g) - np.mean(accel_g)) * 9.81
        
        # 2. Integration to Velocity (mm/s)
        v_m_s = cumulative_trapezoid(a_ms2, t_sec, initial=0)
        v_mm_s = v_m_s * 1000
        
        # 3. Calculate RMS
        v_rms = np.sqrt(np.mean(v_mm_s**2))
        
        # 4. Severity Classification
        status = self._classify(v_rms)
        
        return v_rms, status, v_mm_s

    def _classify(self, v_rms):
        labels = ["GOOD", "SATISFACTORY", "UNSATISFACTORY", "UNACCEPTABLE"]
        for i, limit in enumerate(self.limits):
            if v_rms <= limit:
                return labels[i]
        return labels[-1]