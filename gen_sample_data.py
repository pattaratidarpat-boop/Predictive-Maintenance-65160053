"""สร้างข้อมูลตัวอย่างจำลอง .txt format เหมือนของจริง"""
import numpy as np
import os

def write_waveform_file(path, machine, point, date_str, base_rms=2.0, noise=0.3):
    np.random.seed(abs(hash(path)) % 2**31)
    n = 3000
    dt = 400.0 / n          # ms per sample
    fs = 1000.0 / dt        # Hz
    t = np.linspace(0, 400, n)

    # Target velocity RMS = base_rms mm/s
    # v(t) = V * sin(2π f t)  → v_rms = V/√2
    # a(t) = V * 2πf * cos(2π f t)  → a_rms = V * 2πf / √2
    freq = 25 + np.random.uniform(-3, 3)  # Hz
    omega = 2 * np.pi * freq
    V_peak = base_rms * np.sqrt(2)         # mm/s peak
    A_peak = V_peak * omega / 1000         # m/s²  peak  (V in mm/s → /1000)
    A_g    = A_peak / 9.81                 # convert to G

    accel = A_g * np.cos(omega * t / 1000) + np.random.normal(0, noise * A_g * 0.1, n)

    lines = [
        "                    Waveform Amplitudes",
        "                    *******************",
        f"    Equipment:        {machine}",
        f"    Meas. Point:  {point}    -NAA -->  Motor Outboard Axial",
        f"    Date/Time:  {date_str}    Amplitude:  Acceleration in G-s",
        "", "", "", "", "", "", "", "", "", "",
        "Time (mS) Amplitude Time (mS) Amplitude Time (mS) Amplitude Time (mS) Amplitude",
        "--------- --------- --------- --------- --------- --------- --------- ---------",
    ]
    half = n // 2
    for i in range(half):
        t1, a1 = t[i], accel[i]
        t2, a2 = t[i + half], accel[i + half]
        lines.append(f"{t1:10.3f} {a1:10.3f} {t2:10.3f} {a2:10.3f}")

    with open(path, "w") as f:
        f.write("\n".join(lines))

machines = [
    ("Motor_Compressor_OAH-06_A", "CH-06-A"),
    ("Motor_Compressor_OAH-07_B", "CH-07-B"),
    ("Jockey_Pump_M1A",           "CH-10-A"),
    ("Cooling_Fan_CF-01",         "CH-15-A"),
]

months = ["Jan24", "Feb24", "Mar24", "Apr24", "May24", "Jun24",
          "Jul24", "Aug24", "Sep24", "Oct24", "Nov24", "Dec24"]

# รูปแบบ degradation ต่างกันแต่ละเครื่อง
trends = {
    "Motor_Compressor_OAH-06_A": [1.2, 1.4, 1.5, 1.7, 1.9, 2.1, 2.5, 3.0, 4.2, 5.8, 7.5, 8.2],
    "Motor_Compressor_OAH-07_B": [1.0, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6],
    "Jockey_Pump_M1A":           [2.0, 2.2, 2.5, 3.0, 3.8, 4.5, 5.2, 5.8, 6.5, 7.0, 7.8, 8.5],
    "Cooling_Fan_CF-01":         [3.5, 3.2, 3.8, 4.0, 3.9, 4.2, 4.5, 4.3, 4.8, 5.0, 4.9, 5.2],
}

out = "raw/data"
for mname, point in machines:
    rms_list = trends[mname]
    for i, month in enumerate(months):
        fname = f"A_{mname}__{month}.txt"
        write_waveform_file(
            os.path.join(out, fname),
            mname, point,
            f"16-Oct-24 12:12:21",
            base_rms=rms_list[i]
        )
        print(f"  created: {fname}")

print(f"\n✅ สร้างไฟล์ตัวอย่างเสร็จ ({len(machines) * len(months)} ไฟล์)")
