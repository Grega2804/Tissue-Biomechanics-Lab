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


# #plotting by claude
# fig, ax = plt.subplots(figsize=(10, 7))

# # Plot stress-strain curve
# ax.plot(strain_hold * 100, stress_hold, linewidth=2.5, 
#         color='#2563eb', label='ACL Tensile Test')

# # Styling
# ax.set_xlabel('Strain (%)', fontsize=13, fontweight='bold')
# ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
# ax.set_title('Stress-Strain Diagram - Human ACL Quasi-Static Tensile Test', 
#              fontsize=14, fontweight='bold', pad=20)

# # Grid
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
# ax.set_axisbelow(True)

# # Legend
# ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')

# # Improve tick labels
# ax.tick_params(axis='both', which='major', labelsize=11)

# # Add subtle background color
# ax.set_facecolor('#f8f9fa')
# fig.patch.set_facecolor('white')

# # Tight layout
# plt.tight_layout()

# # Display
# plt.show()

# Optional: Print key mechanical properties
max_stress = np.max(stress_hold)
max_strain = strain_hold[np.argmax(stress_hold)] * 100

# print(f"\n{'='*50}")
# print(f"Mechanical Properties Summary")
# print(f"{'='*50}")
# print(f"Maximum Stress: {max_stress:.2f} MPa")
# print(f"Strain at Max Stress: {max_strain:.2f} %")
# print(f"Gauge Length: {length_hold:.2f} mm")
# print(f"Cross-sectional Area: {area_hold:.2f} mm²")
# print(f"{'='*50}\n")





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

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 7))

# # Plot stress over time
# ax.plot(time_filtered, stress_filtered, linewidth=2.5, 
#         color='#dc2626', marker='o', markersize=4, 
#         markerfacecolor='white', markeredgewidth=1.5,
#         label='Stress at 1 mm Displacement')

# # Styling
# ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
# ax.set_ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
# ax.set_title('Stress Relaxation at Constant Displacement (1 mm)', 
#              fontsize=14, fontweight='bold', pad=20)

# # Grid
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
# ax.set_axisbelow(True)

# # Legend
# ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')

# # Improve tick labels
# ax.tick_params(axis='both', which='major', labelsize=11)

# # Add subtle background color
# ax.set_facecolor('#f8f9fa')
# fig.patch.set_facecolor('white')

# # Tight layout
# plt.tight_layout()

# # Display
# plt.show()

# Print summary statistics
initial_stress = stress_filtered[0]
final_stress = stress_filtered[-1]
stress_drop = initial_stress - final_stress
stress_drop_percent = (stress_drop / initial_stress) * 100

# print(f"\n{'='*50}")
# print(f"Stress Relaxation Summary (at 1 mm displacement)")
# print(f"{'='*50}")
# print(f"Number of data points: {len(time_filtered)}")
# print(f"Time range: {time_filtered[0]:.2f} - {time_filtered[-1]:.2f} s")
# print(f"Initial stress: {initial_stress:.2f} MPa")
# print(f"Final stress: {final_stress:.2f} MPa")
# print(f"Stress drop: {stress_drop:.2f} MPa ({stress_drop_percent:.1f}%)")
# print(f"{'='*50}\n")


from scipy.optimize import curve_fit

# --- SETUP: Time-shifting and Model Definition ---

# It is good practice to start time from 0 for decay fitting
time_fit = time_filtered - time_filtered[0]

# Model for a two-term exponential decay (viscoelastic stress relaxation)
# Stress(t) = C_eq + A1*exp(R1*t) + A2*exp(R2*t)
def two_term_exp_model(t, C_eq, A1, R1, A2, R2):
    return C_eq + A1 * np.exp(R1 * t) + A2 * np.exp(R2 * t)

# --- HIGH-QUALITY INITIAL GUESSES (Derived from source Mathematica code) ---

# We base the guesses on the two most dominant Maxwell elements from the source.
guess_C_eq = stress_filtered[-1]        # Equilibrium is the final stress value.
guess_A1 = 3.5                          # Amplitude of the fast decay.
guess_R1 = -20.0                        # Rate of the fast decay (-1 / 0.049s).
guess_A2 = 1.8                          # Amplitude of the slow decay.
guess_R2 = -2.1                         # Rate of the slow decay (-1 / 0.478s).

p0_derived = [guess_C_eq, guess_A1, guess_R1, guess_A2, guess_R2]

# Print the derived guesses for inspection
print("\n" + "="*50)
print("Using High-Quality Initial Guesses Derived from Source:")
print(f"  p0 = {p0_derived}")
print("="*50 + "\n")


# Define physical bounds to keep the fit stable
bounds = (
    [0, 0, -np.inf, 0, -np.inf],      # Lower bounds for [C_eq, A1, R1, A2, R2]
    [np.inf, np.inf, 0, np.inf, 0]    # Upper bounds
)

# --- PERFORM THE CURVE FIT ---
try:
    params, covariance = curve_fit(
        two_term_exp_model,
        time_fit,
        stress_filtered,
        p0=p0_derived,    # Use our new, highly-informed guesses
        bounds=bounds,
        maxfev=5000
    )

    # Print the final optimized parameters
    print("\n" + "="*50)
    print("Final Fitted Parameters for Two-Term Model")
    print("="*50)
    print(f"Equilibrium Stress (C_eq): {params[0]:.3f} MPa")
    print(f"Amplitude 1 (A1):          {params[1]:.3f} MPa")
    print(f"Decay Rate 1 (R1):         {params[2]:.3f}")
    print(f"Amplitude 2 (A2):          {params[3]:.3f} MPa")
    print(f"Decay Rate 2 (R2):         {params[4]:.3f}")
    print("="*50 + "\n")

    # Generate the fitted curve using the optimized parameters
    stress_fit_final = two_term_exp_model(time_fit, *params)
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    plt.plot(time_filtered, stress_filtered, 'o', color='#d9534f', markersize=6, label='Experimental Data')
    plt.plot(time_filtered, stress_fit_final, '-', color='#0275d8', linewidth=3, label='Two-Term Exponential Fit')
    
    # Styling
    plt.xlabel('Time (s)', fontsize=13, fontweight='bold')
    plt.ylabel('Stress (MPa)', fontsize=13, fontweight='bold')
    plt.title('High-Quality Two-Term Exponential Fit of Stress Relaxation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

except RuntimeError as e:
    print(f"ERROR: The curve fitting failed to converge. Reason: {e}")



#Import 
from scipy.optimize import curve_fit


#time_filtered = time_filtered - 0.1
#print(time_filtered)

# Exponential function model
def test_eq(x, a, b, c):
    return a * np.exp(b * x) + c

param, param_cov = curve_fit(test_eq, time_filtered, stress_filtered, p0=[stress_filtered[0]-stress_filtered[-1], -0.1 , stress_filtered[-1]])

# Print optimized parameters and their covariance
# print("Exponential function coefficients:")
# print(param)
# print("Covariance of coefficients:")
# print(param_cov)


#generating fitted y values
ans = param[0] *  np.exp(param[1] * time_filtered) + param[2]


# plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
# plt.plot(time_filtered, ans, '--', color='blue', label='Fitted curve')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Exponential Curve Fitting')
# plt.legend()
# plt.show()

#The code probably has different Maxwell branches, like in the provided sheet
#So lets try some higher order multiple curve fits


def test_eq2(x, a, b, c, d, e):
    return c + a * np.exp(b * x) + d * np.exp(e * x)

# p2 = [stress_filtered[-1],  # long-term equilibrium stress
#       stress_filtered[0] - stress_filtered[-1],  # total amplitude
#       0.5,  # tau1 fast
#       (stress_filtered[0] - stress_filtered[-1])/2, 5.0]
#Shitty guess, find a better one

#--- Constructing Better Initial Guesses ---

# C_eq: The long-term equilibrium stress is the final stress in the data.
initial_guess_c = stress_filtered[-1]

# The total stress drop is the initial stress minus the final stress.
total_stress_drop = stress_filtered[0] - stress_filtered[-1]

# A1, A2: Let's guess the two amplitudes each contribute to half of the total drop.
initial_guess_A1 = total_stress_drop / 2.0
initial_guess_A2 = total_stress_drop / 2.0

# R1, R2: The decay rates MUST be negative. Let's guess one is for a "fast" decay
# and the other for a "slow" decay.
initial_guess_R1 = -1.0  # Fast decay rate
initial_guess_R2 = -0.1  # Slow decay rate

p2= [initial_guess_c, initial_guess_A1, initial_guess_R1, initial_guess_A2, initial_guess_R2]







param2, param_cov2 = curve_fit(test_eq2, time_filtered, stress_filtered, p2)

ans2 = param2[0] *  np.exp(param2[1] * time_filtered) + param2[2] +param2[3]*np.exp(param2[4] * time_filtered)


plt.plot(time_filtered, stress_filtered, 'o', color='red', label='Experimental data')
plt.plot(time_filtered, ans2, '--', color='blue', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Curve Fitting')
plt.legend()
plt.show()