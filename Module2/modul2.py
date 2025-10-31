from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

os.chdir(Path(__file__).parent)

def parse_txt(path):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    
    for line in lines:
        if line.startswith("Gauge length"):
            length = float(line.split(":")[1].strip())
        elif line.startswith("Area"):
            area = float(line.split(":")[1].strip())
    
    i_results = lines.index("Time [s],Displacement [mm],Force [N]")
    results = lines[i_results+1:]
    
    time, displacement, force = [], [], []
    for i in results:
        split = i.split(",")
        time.append(float(split[0]))
        displacement.append(float(split[1]))
        force.append(float(split[2]))
    
    return length, area, time, displacement, force

def plot_stress_strain(strain, stress, title, color='#2563eb'):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(strain * 100, stress, linewidth=2.5, color=color, label='ACL Tensile Test')
    
    ax.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()

def print_properties(max_stress, max_strain, length, area):
    print(f"\n{'='*50}")
    print(f"Mechanical Properties Summary")
    print(f"{'='*50}")
    print(f"Maximum Stress: {max_stress:.2f} MPa")
    print(f"Strain at Max Stress: {max_strain:.2f} %")
    print(f"Gauge Length: {length:.2f} mm")
    print(f"Cross-sectional Area: {area:.2f} mm²")
    print(f"{'='*50}\n")

def create_exp_function(n_exp):
    def func(x, c, *params):
        result = c
        for i in range(n_exp):
            a = params[2*i]
            b = params[2*i + 1]
            result += a * np.exp(b * x)
        return result
    return func

def create_bounds(n_exp):
    lower = [0]
    upper = [np.inf]
    
    for i in range(n_exp):
        lower.extend([0, -np.inf])
        upper.extend([np.inf, 0])
    
    return (lower, upper)

def calculate_fitted_values(x, params, n_exp):
    c = params[0]
    result = c + np.zeros_like(x)
    for i in range(n_exp):
        a = params[2*i + 1]
        b = params[2*i + 2]
        result += a * np.exp(b * x)
    return result


path_hold = Path("Hold.txt")
length_hold, area_hold, time_hold, displacement_hold, force_hold = parse_txt(path_hold)

force_hold = np.array(force_hold)
displacement_hold = np.array(displacement_hold)
stress_hold = force_hold / area_hold
strain_hold = displacement_hold / length_hold

plot_stress_strain(strain_hold, stress_hold, 'Stress-Strain Diagram - Human ACL Quasi-Static Tensile Test')

max_stress = np.max(stress_hold)
max_strain = strain_hold[np.argmax(stress_hold)] * 100
print_properties(max_stress, max_strain, length_hold, area_hold)


path_sin = Path("HalfSecSinus.txt")
length_sin, area_sin, time_sin, displacement_sin, force_sin = parse_txt(path_sin)

force_sin = np.array(force_sin)
displacement_sin = np.array(displacement_sin)
stress_sin = force_sin / area_sin
strain_sin = displacement_sin / length_sin

plot_stress_strain(strain_sin, stress_sin, 'Stress-Strain Diagram - Human ACL 0.5 Sec Period')

max_stress_sin = np.max(stress_sin)
max_strain_sin = strain_sin[np.argmax(stress_hold)] * 100
print_properties(max_stress_sin, max_strain_sin, length_sin, area_sin)


path_decay = Path("Decay.txt")
length_decay, area_decay, time_decay, displacement_decay, force_decay = parse_txt(path_decay)

time_decay = np.array(time_decay)
force_decay = np.array(force_decay)
displacement_decay = np.array(displacement_decay)
stress_decay = force_decay / area_decay

tolerance = 0.00001
mask = np.abs(displacement_decay - 1.0) < tolerance
time_filtered = time_decay[mask]
stress_filtered = stress_decay[mask]

fig, ax = plt.subplots(figsize=(10, 7))
plt.xscale('log')
ax.plot(time_filtered, stress_filtered, linewidth=2.5, color='#dc2626', 
        marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5,
        label='Stress at 1 mm Displacement')

ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
ax.set_title('Stress Relaxation at Constant Displacement (1 mm)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

initial_stress = stress_filtered[0]
final_stress = stress_filtered[-1]
stress_drop = initial_stress - final_stress
stress_drop_percent = (stress_drop / initial_stress) * 100

print(f"\n{'='*50}")
print(f"Stress Relaxation Summary (at 1 mm displacement)")
print(f"{'='*50}")
print(f"Number of data points: {len(time_filtered)}")
print(f"Time range: {time_filtered[0]:.2f} - {time_filtered[-1]:.2f} s")
print(f"Initial stress: {initial_stress:.2f} MPa")
print(f"Final stress: {final_stress:.2f} MPa")
print(f"Stress drop: {stress_drop:.2f} MPa ({stress_drop_percent:.1f}%)")
print(f"{'='*50}\n")


n_exp_terms = []
rmse_values = []
params_dict = {}
fitted_curves = {}
prev_rmse = np.inf
stopped_early = False

for n in range(1, 11):
    print(f"\n{'='*50}")
    print(f"Testing model with {n} exponential terms")
    print(f"{'='*50}")
    
    try:
        func = create_exp_function(n)
        bounds = create_bounds(n)
        params, param_cov = curve_fit(func, time_filtered, stress_filtered, 
                                      bounds=bounds, maxfev=100000000)
        
        fitted = calculate_fitted_values(time_filtered, params, n)
        rmse = np.sqrt(np.mean((stress_filtered - fitted)**2))
        
        n_exp_terms.append(n)
        rmse_values.append(rmse)
        params_dict[n] = params
        fitted_curves[n] = fitted
        
        print(f"RMSE: {rmse:.6f}")
        print(f"Parameters: c = {params[0]:.4f}")
        for i in range(n):
            print(f"  a{i+1} = {params[2*i+1]:.4f}, b{i+1} = {params[2*i+2]:.4f}")
        
        if rmse > prev_rmse:
            print(f"\n*** RMSE increased from {prev_rmse:.6f} to {rmse:.6f} ***")
            print(f"*** Stopping iteration (model with {n} terms is worse) ***")
            stopped_early = True
            break
        
        prev_rmse = rmse
        
    except Exception as e:
        print(f"Failed to converge for {n} exponential terms: {str(e)}")
        break

plt.figure(figsize=(10, 6))
plt.plot(n_exp_terms, rmse_values, 'o-', linewidth=2, markersize=8)

if stopped_early and len(n_exp_terms) > 1:
    plt.plot(n_exp_terms[-1], rmse_values[-1], 'ro', markersize=12, 
             label='Stopping point (RMSE increased)', zorder=5)
    plt.axhline(y=rmse_values[-2], color='g', linestyle='--', alpha=0.5, 
                label=f'Previous best RMSE: {rmse_values[-2]:.6f}')

plt.xlabel('Number of Exponential Terms', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('RMSE vs Model Complexity', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(n_exp_terms)
if stopped_early:
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_filtered, stress_filtered, 'o', color='red', 
         label='Experimental data', markersize=4, alpha=0.6)

colors = plt.cm.viridis(np.linspace(0, 1, len(fitted_curves)))
for idx, (n, fitted) in enumerate(fitted_curves.items()):
    linestyle = '--' if idx < len(fitted_curves) - 1 else ':'
    linewidth = 1.5 if idx < len(fitted_curves) - 1 else 2.5
    alpha = 1.0 if idx < len(fitted_curves) - 1 else 0.7
    
    label_text = f'n={n} (RMSE={rmse_values[idx]:.4f})'
    if idx == len(fitted_curves) - 1 and stopped_early:
        label_text += ' [Worse]'
    
    plt.plot(time_filtered, fitted, linestyle=linestyle, color=colors[idx], 
             label=label_text, linewidth=linewidth, alpha=alpha)

plt.xlabel('Time', fontsize=12)
plt.ylabel('Stress', fontsize=12)
plt.xscale('log')
title = 'Curve Fitting: All Models'
if stopped_early:
    title += f' (Stopped at n={n_exp_terms[-1]})'
plt.title(title, fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for n, rmse in zip(n_exp_terms, rmse_values):
    marker = " [WORSE - STOPPED HERE]" if (n == n_exp_terms[-1] and stopped_early) else ""
    print(f"n={n}: RMSE = {rmse:.6f}{marker}")

if stopped_early and len(rmse_values) > 1:
    best_idx = len(rmse_values) - 2
    print(f"\nBest model: {n_exp_terms[best_idx]} exponential terms")
    print(f"Minimum RMSE: {rmse_values[best_idx]:.6f}")
    print(f"(Stopped at n={n_exp_terms[-1]} because RMSE increased to {rmse_values[-1]:.6f})")
else:
    best_idx = np.argmin(rmse_values)
    print(f"\nBest model: {n_exp_terms[best_idx]} exponential terms")
    print(f"Minimum RMSE: {rmse_values[best_idx]:.6f}")