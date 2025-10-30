#File to check the data for module2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import os
# Always look for files in the same folder as the script
os.chdir(Path(__file__).parent)

#Function for reading in the log-files
def parse_txt(path):
    
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    #get the gauge_length and 
    for line in lines:
        if line.startswith("Gauge length"):
            length = float(line.split(":")[1].strip())
        elif line.startswith("Area"):
            area = float(line.split(":")[1].strip())
            
    #get all the results
    i_results = lines.index("Time [s],Displacement [mm],Force [N]")
    results = lines[i_results+1:]
    time = []
    displacement = []
    force = []
    for i in results:
        split = i.split(",")
        time.append(float(split[0]))
        displacement.append(float(split[1]))
        force.append(float(split[2]))
    return length, area, time, displacement, force


        
#Firstly: Identify the quasi-static tensile elastic behaviour of the human anterior cruciate ligament (ACL). Are the resulting deformations reversible
path_hold = Path("Hold.txt")
length_hold, area_hold, time_hold, displacement_hold, force_hold = parse_txt(path_hold)
#Convert to numpy arrays for easier handling

force_hold = np.array(force_hold) 
displacement_hold = np.array(displacement_hold)
stress_hold = force_hold/area_hold
strain_hold = displacement_hold/length_hold


#plotting by claude
fig, ax = plt.subplots(figsize=(10, 7))

# Plot stress-strain curve
ax.plot(strain_hold * 100, stress_hold, linewidth=2.5, 
        color='#2563eb', label='ACL Tensile Test')

# Styling
ax.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
ax.set_title('Stress-Strain Diagram - Human ACL Quasi-Static Tensile Test', 
             fontsize=14, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=11)

# Add subtle background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Tight layout
plt.tight_layout()

# Display
plt.show()

# Optional: Print key mechanical properties
max_stress = np.max(stress_hold)
max_strain = strain_hold[np.argmax(stress_hold)] * 100

print(f"\n{'='*50}")
print(f"Mechanical Properties Summary")
print(f"{'='*50}")
print(f"Maximum Stress: {max_stress:.2f} MPa")
print(f"Strain at Max Stress: {max_strain:.2f} %")
print(f"Gauge Length: {length_hold:.2f} mm")
print(f"Cross-sectional Area: {area_hold:.2f} mm²")
print(f"{'='*50}\n")





#For point 3
#Firstly: Identify the quasi-static tensile elastic behaviour of the human anterior cruciate ligament (ACL). Are the resulting deformations reversible
path_decay = Path("Decay.txt")
length_decay, area_decay, time_decay, displacement_decay, force_decay = parse_txt(path_decay)
#Convert to numpy arrays for easier handling

time_decay = np.array(time_decay)
force_decay = np.array(force_decay)
displacement_decay = np.array(displacement_decay)

# Calculate stress
stress_decay = force_decay / area_decay

# Filter for displacement exactly equal to 1 mm
# Using a small tolerance for floating point comparison
tolerance = 0.00001
mask = np.abs(displacement_decay - 1.0) < tolerance

time_filtered = time_decay[mask]
stress_filtered = stress_decay[mask]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot stress over time
ax.plot(time_filtered, stress_filtered, linewidth=2.5, 
        color='#dc2626', marker='o', markersize=4, 
        markerfacecolor='white', markeredgewidth=1.5,
        label='Stress at 1 mm Displacement')

# Styling
ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
ax.set_title('Stress Relaxation at Constant Displacement (1 mm)', 
             fontsize=14, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=11)

# Add subtle background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Tight layout
plt.tight_layout()

# Display
plt.show()

# Print summary statistics
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

#Import 
from scipy.optimize import curve_fit


#time_filtered = time_filtered - 0.1
#print(time_filtered)

# Exponential function model
def test_eq(x, a, b, c):
    return a * np.exp(b * x) + c

param, param_cov = curve_fit(test_eq, time_filtered, stress_filtered, p0=[stress_filtered[0]-stress_filtered[-1], -0.1 , stress_filtered[-1]])

#Print optimized parameters and their covariance
print("Exponential function coefficients:")
print(param)
print("Covariance of coefficients:")
print(param_cov)


#generating fitted y values
ans = param[0] *  np.exp(param[1] * time_filtered) + param[2]

rmse = np.sqrt(np.mean((stress_filtered - ans)**2))
print(f"RMSE: {rmse}")


plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Curve Fitting')
plt.legend()
plt.show()

#The code probably has different Maxwell branches, like in the provided sheet
#So lets try some higher order multiple curve fits
def test_eq2(x, c, a1, b1, a2, b2):
    return c + a1  * np.exp(b1 * x) + a2 * np.exp(b2 * x)


bound2 = ([0, 0, -np.inf, 0, -np.inf],      # bounds from theory, b<0 and a>0, is needed
    [np.inf, np.inf, 0, np.inf, 0])
param2, param_cov2 = curve_fit(test_eq2, time_filtered, stress_filtered, bounds=bound2, maxfev=1000000)
ans2 = param2[0] + param2[1]*np.exp(param2[2]*time_filtered) + param2[3]*np.exp(param2[4]*time_filtered)

rmse2 = np.sqrt(np.mean((stress_filtered - ans2)**2))
print(f"RMSE: {rmse2}")


plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans2, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting')
plt.legend()
plt.show()
#Showed convergence, try for n=3



def test_eq3(x, c, a1, b1, a2, b2, a3, b3):
    return c + a1  * np.exp(b1 * x) + a2 * np.exp(b2 * x) + a3 * np.exp(b3 * x)


bound3= ([0, 0, -np.inf, 0, -np.inf, 0, -np.inf],      # bounds from theory, b<0 and a>0, is needed
    [np.inf, np.inf, 0, np.inf, 0,np.inf, 0])
param3, param_cov3 = curve_fit(test_eq3, time_filtered, stress_filtered, bounds=bound3, maxfev=1000000)
ans3 = param3[0] + param3[1]*np.exp(param3[2]*time_filtered) + param3[3]*np.exp(param3[4]*time_filtered) + param3[5]*np.exp(param3[6]*time_filtered)

rmse3 = np.sqrt(np.mean((stress_filtered - ans3)**2))
print(f"RMSE: {rmse3}")


plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans3, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting')
plt.legend()
plt.show()
#again convergenve, try 4


def test_eq4(x, c, a1, b1, a2, b2, a3, b3, a4, b4):
    return c + a1  * np.exp(b1 * x) + a2 * np.exp(b2 * x) + a3 * np.exp(b3 * x) + a4 * np.exp(b4 * x)

bound4= ([0, 0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf],      # bounds from theory, b<0 and a>0, is needed
    [np.inf, np.inf, 0, np.inf, 0, np.inf, 0, np.inf, 0])
param4, param_cov4 = curve_fit(test_eq4, time_filtered, stress_filtered, bounds=bound4, maxfev=1000000)
ans4 = param4[0] + param4[1]*np.exp(param4[2]*time_filtered) + param4[3]*np.exp(param4[4]*time_filtered) + param4[5]*np.exp(param4[6]*time_filtered) + param4[7]*np.exp(param4[8]*time_filtered)

rmse4 = np.sqrt(np.mean((stress_filtered - ans4)**2))
print(f"RMSE: {rmse4}")

plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans4, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting')
plt.legend()
plt.show()
#convergence, try 5 

def test_eq5(x, c, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5):
    return c + a1  * np.exp(b1 * x) + a2 * np.exp(b2 * x) + a3 * np.exp(b3 * x) + a4 * np.exp(b4 * x)+ a5 * np.exp(b5 * x)

bound5= ([0, 0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf],      # bounds from theory, b<0 and a>0, is needed
    [np.inf, np.inf, 0, np.inf, 0, np.inf, 0, np.inf, 0, np.inf, 0])
param5, param_cov5 = curve_fit(test_eq5, time_filtered, stress_filtered, bounds=bound5, maxfev=1000000)
ans5 = param5[0] + param5[1]*np.exp(param5[2]*time_filtered) + param5[3]*np.exp(param5[4]*time_filtered) + param5[5]*np.exp(param5[6]*time_filtered) + param5[7]*np.exp(param5[8]*time_filtered) + param5[9]*np.exp(param5[10]*time_filtered)

rmse5 = np.sqrt(np.mean((stress_filtered - ans5)**2))
print(f"RMSE: {rmse5}")

plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans5, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting')
plt.legend()
plt.show()

print(param4)
print(param5)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming time_filtered and stress_filtered are already defined
# If not, uncomment and define them:
# time_filtered = np.array([...])
# stress_filtered = np.array([...])

def create_exp_function(n_exp):
    """Create an exponential function with n exponential terms"""
    def func(x, c, *params):
        result = c
        for i in range(n_exp):
            a = params[2*i]
            b = params[2*i + 1]
            result += a * np.exp(b * x)
        return result
    return func

def create_bounds(n_exp):
    """Create bounds for n exponential terms"""
    lower = [0]  # c >= 0
    upper = [np.inf]  # c <= inf
    
    for i in range(n_exp):
        lower.extend([0, -np.inf])  # a >= 0, b can be any value but expect b < 0
        upper.extend([np.inf, 0])   # a <= inf, b <= 0
    
    return (lower, upper)

def calculate_fitted_values(x, params, n_exp):
    """Calculate fitted values given parameters"""
    c = params[0]
    result = c + np.zeros_like(x)
    for i in range(n_exp):
        a = params[2*i + 1]
        b = params[2*i + 2]
        result += a * np.exp(b * x)
    return result

# Storage for results
n_exp_terms = []
rmse_values = []
params_dict = {}
fitted_curves = {}

# Iterate from 2 to 10 exponential terms
prev_rmse = np.inf
stopped_early = False

for n in range(2, 11):
    print(f"\n{'='*50}")
    print(f"Testing model with {n} exponential terms")
    print(f"{'='*50}")
    
    try:
        # Create function and bounds
        func = create_exp_function(n)
        bounds = create_bounds(n)
        
        # Fit the curve
        params, param_cov = curve_fit(func, time_filtered, stress_filtered, 
                                      bounds=bounds, maxfev=1000000)
        
        # Calculate fitted values
        fitted = calculate_fitted_values(time_filtered, params, n)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((stress_filtered - fitted)**2))
        
        # Store results
        n_exp_terms.append(n)
        rmse_values.append(rmse)
        params_dict[n] = params
        fitted_curves[n] = fitted
        
        print(f"RMSE: {rmse:.6f}")
        print(f"Parameters: c = {params[0]:.4f}")
        for i in range(n):
            print(f"  a{i+1} = {params[2*i+1]:.4f}, b{i+1} = {params[2*i+2]:.4f}")
        
        # Check if RMSE got worse
        if rmse > prev_rmse:
            print(f"\n*** RMSE increased from {prev_rmse:.6f} to {rmse:.6f} ***")
            print(f"*** Stopping iteration (model with {n} terms is worse) ***")
            stopped_early = True
            break
        
        prev_rmse = rmse
        
    except Exception as e:
        print(f"Failed to converge for {n} exponential terms: {str(e)}")
        break

# Plot RMSE vs number of exponential terms
plt.figure(figsize=(10, 6))
plt.plot(n_exp_terms, rmse_values, 'o-', linewidth=2, markersize=8)

# Highlight the last point if stopped early
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

# Plot all fitted curves
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
title = 'Curve Fitting: All Models'
if stopped_early:
    title += f' (Stopped at n={n_exp_terms[-1]})'
plt.title(title, fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for n, rmse in zip(n_exp_terms, rmse_values):
    marker = " [WORSE - STOPPED HERE]" if (n == n_exp_terms[-1] and stopped_early) else ""
    print(f"n={n}: RMSE = {rmse:.6f}{marker}")

# Find best model (excluding the last one if it's worse)
if stopped_early and len(rmse_values) > 1:
    best_idx = len(rmse_values) - 2  # Second to last
    print(f"\nBest model: {n_exp_terms[best_idx]} exponential terms")
    print(f"Minimum RMSE: {rmse_values[best_idx]:.6f}")
    print(f"(Stopped at n={n_exp_terms[-1]} because RMSE increased to {rmse_values[-1]:.6f})")
else:
    best_idx = np.argmin(rmse_values)
    print(f"\nBest model: {n_exp_terms[best_idx]} exponential terms")
    print(f"Minimum RMSE: {rmse_values[best_idx]:.6f}")
