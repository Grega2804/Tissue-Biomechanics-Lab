import re
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression


# === Settings ===
TXT_PATH = Path("file.txt")
SAVE_DIR = Path(".")

n_ramp_up = 50 #change this if ramp up is changed

# === Parsing ===
def parse_txt(path):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    i_sample = lines.index("SAMPLE PARAMETERS")
    i_sched = lines.index("SCHEDULE PARAMETERS")
    i_results = lines.index("RESULTS")
    
    sample = lines[i_sample+1:i_sched]
    results = lines[i_results+1:]
    #print(results)
    #print(sample)
    gauge_lengths, mean_diams = {}, {}
    for ln in sample:
        if m := re.match(r"Gauge length\s+(\d+)\s*,\s*([\deE\.\+\-]+)", ln):
            gauge_lengths[int(m.group(1))] = float(m.group(2))
        elif m := re.match(r"Mean diameter\s+(\d+)\s*,\s*([\deE\.\+\-]+)", ln):
            mean_diams[int(m.group(1))] = float(m.group(2))
    
    header = [h.strip() for h in results[0].split(",")]
    i_force = next(i for i, c in enumerate(header) if c.startswith("Force"))
    i_disp = {j: next(i for i, c in enumerate(header) if c.startswith(f"Disp {j}")) for j in range(1, 6)}
    
    arr = np.array([list(map(float, ln.split(","))) for ln in results[1:]])
    force = arr[:, i_force]
    disp = {j: arr[:, i_disp[j]] for j in range(1, 6)}
    return gauge_lengths, mean_diams, disp, force



# === Main ===
gauge_lengths, mean_diams, disp, force = parse_txt(TXT_PATH)

# Plot all gauges together
plt.figure(figsize=(8, 5))
modulus = []
for i in range(1, 5):
    L = gauge_lengths[i]
    d = mean_diams[i]
    A = math.pi * (d ** 2) / 4
    strain = (disp[i+1] - disp[i]) / L
    stress = force / A
    plt.plot(strain, stress, label=f"Gauge {i}-{i+1}")
    #differnent modulus in the gauge sections
    strain_ru = strain[:n_ramp_up]
    stress_ru = stress[:n_ramp_up]
    model = LinearRegression()
    model.fit(strain_ru.reshape(-1, 1), stress_ru)
    modulus.append(model.coef_[0])
    
    

    

plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
plt.title("Stress-Strain Curves (All Gauges)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(SAVE_DIR / "stress_strain_all_gauges.png", dpi=200)
plt.show()

# Total probe
L_total = sum(gauge_lengths.values())
d_mean = np.mean(list(mean_diams.values()))
A_mean = math.pi * (d_mean ** 2) / 4
strain_total = (disp[5] - disp[1]) / L_total
stress_total = force / A_mean

plt.figure(figsize=(8, 5))
plt.plot(strain_total, stress_total, color="black", linewidth=2)
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
plt.title("Stress-Strain (Total Probe 1-5)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(SAVE_DIR / "stress_strain_total.png", dpi=200)
plt.show()

#Next check the tensile modulus
#I will do this firstly for the whole probe
strain_ru_total = strain_total[:n_ramp_up]
stress_ru_total = stress_total[:n_ramp_up]
#plt.figure(figsize=(8, 5))
#plt.plot(strain_ru, stress_ru, color="black", linewidth=2)
#plt.show()

# Fit linear regression
#Using machine learing library
model = LinearRegression()
model.fit(strain_ru_total.reshape(-1, 1), stress_ru_total)

# Get the gradient (Young's modulus)
gradient = model.coef_[0]

print(f"Gradient (Young's modulus): {gradient:.2f}")
#print(L_total)

#Now checking the R-squared error to see how good the line is
r2 = model.score(strain_ru_total.reshape(-1, 1), stress_ru_total)
print(r2)
print(modulus)