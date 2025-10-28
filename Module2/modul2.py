#File to check the data for module2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
tolerance = 0.001
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

from scipy.optimize import curve_fit

# Define exponential decay model for stress relaxation
# σ(t) = σ_eq + (σ_0 - σ_eq) * exp(-t/τ)
# where σ_eq is equilibrium stress, σ_0 is initial stress, τ is relaxation time
def stress_relaxation_model(t, sigma_eq, sigma_0, tau):
    """
    Exponential stress relaxation model
    t: time
    sigma_eq: equilibrium stress (long-term stress)
    sigma_0: initial stress
    tau: relaxation time constant
    """
    return sigma_eq + (sigma_0 - sigma_eq) * np.exp(-t / tau)

# Shift time to start at 0 for better fitting
time_fit = time_filtered - time_filtered[0]

# Initial parameter guesses
sigma_0_guess = stress_filtered[0]
sigma_eq_guess = stress_filtered[-1]
tau_guess = 50  # Initial guess for relaxation time in seconds

# Perform curve fitting
popt, pcov = curve_fit(stress_relaxation_model, time_fit, stress_filtered,
                       p0=[sigma_eq_guess, sigma_0_guess, tau_guess],
                       maxfev=10000)

sigma_eq_fit, sigma_0_fit, tau_fit = popt

# Calculate R-squared
residuals = stress_filtered - stress_relaxation_model(time_fit, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((stress_filtered - np.mean(stress_filtered))**2)
r_squared = 1 - (ss_res / ss_tot)

# Generate fitted curve
time_dense = np.linspace(0, time_fit[-1], 500)
stress_fit = stress_relaxation_model(time_dense, *popt)

# Create the plot with fit
fig, ax = plt.subplots(figsize=(10, 7))

# Plot experimental data
ax.plot(time_filtered, stress_filtered, 'o', markersize=6, 
        color='#dc2626', markerfacecolor='white', markeredgewidth=2,
        label='Experimental Data', zorder=3)

# Plot fitted curve
ax.plot(time_filtered[0] + time_dense, stress_fit, linewidth=2.5, 
        color='#2563eb', linestyle='--',
        label=f'Exponential Fit (R² = {r_squared:.4f})', zorder=2)

# Styling
ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
ax.set_title('Stress Relaxation with Exponential Curve Fit', 
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

# Add text box with fit parameters
textstr = f'σ(t) = σ_eq + (σ₀ - σ_eq)·exp(-t/τ)\n\n'
textstr += f'σ₀ = {sigma_0_fit:.2f} MPa\n'
textstr += f'σ_eq = {sigma_eq_fit:.2f} MPa\n'
textstr += f'τ = {tau_fit:.2f} s'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

# Tight layout
plt.tight_layout()
plt.show()

# Print detailed results
print(f"\n{'='*60}")
print(f"Stress Relaxation Curve Fit Results")
print(f"{'='*60}")
print(f"Model: σ(t) = σ_eq + (σ₀ - σ_eq) × exp(-t/τ)")
print(f"{'-'*60}")
print(f"Fitted Parameters:")
print(f"  Initial Stress (σ₀):      {sigma_0_fit:.3f} MPa")
print(f"  Equilibrium Stress (σ_eq): {sigma_eq_fit:.3f} MPa")
print(f"  Relaxation Time (τ):       {tau_fit:.3f} s")
print(f"{'-'*60}")
print(f"Stress Drop:")
print(f"  Absolute: {sigma_0_fit - sigma_eq_fit:.3f} MPa")
print(f"  Relative: {((sigma_0_fit - sigma_eq_fit)/sigma_0_fit)*100:.1f}%")
print(f"{'-'*60}")
print(f"Goodness of Fit:")
print(f"  R² = {r_squared:.6f}")
print(f"  RMSE = {np.sqrt(ss_res/len(stress_filtered)):.4f} MPa")
print(f"{'-'*60}")
print(f"Time to reach 63% relaxation (1τ): {tau_fit:.2f} s")
print(f"Time to reach 95% relaxation (3τ): {3*tau_fit:.2f} s")
print(f"{'='*60}\n")