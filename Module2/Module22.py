# File to check the data for module2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# Always look for files in the same folder as the script
os.chdir(Path(__file__).parent)

# Function for reading in the log-files
def parse_txt(path):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    
    # Get the gauge_length and area
    for line in lines:
        if line.startswith("Gauge length"):
            length = float(line.split(":")[1].strip())
        elif line.startswith("Area"):
            area = float(line.split(":")[1].strip())
    
    # Get all the results
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
    
    # Convert to numpy arrays before returning
    time = np.array(time)
    displacement = np.array(displacement)
    force = np.array(force)
    
    return length, area, time, displacement, force

# Define all paths including Hold.txt
paths = [
    "strain005.txt",
    "strain01.txt",
    "strain02.txt", 
    "strain04.txt",
    "strain08.txt",
    "strain16.txt",
    "strain32.txt",
    "strain64.txt",
    "strain128.txt",
    "strain100sec.txt",
    "strain1000sec.txt"
]

# Initialize dictionaries to store data for each file
data = {}

# Read all files and store as numpy arrays
for path_str in paths:
    path = Path(path_str)
    length, area, time, displacement, force = parse_txt(path)
    
    # Remove the last value from all arrays
    time = time[:-1]
    displacement = displacement[:-1]
    force = force[:-1]
    
    # Store all data including derived quantities
    data[path_str] = {
        'length': length,
        'area': area,
        'time': time,
        'displacement': displacement,
        'force': force,
        'stress': force / area,
        'strain': displacement / length
    }

# Create a simple plot overlaying all strain-stress curves
plt.figure(figsize=(10, 6))


name = [
    "50 ms loading",
    "100 ms Loading",
    "200 ms Loading", 
    "400 ms Loading",
    "800 ms Loading",
    "1600 ms Loading",
    "3200 ms Loading",
    "6400 ms Loading",
    "12800 ms Loading",
    "100 s Loading",
    "1000 s Loading"
]
for i, (filename, dataset) in enumerate(data.items()):
    # Find indices where displacement <= 0.99
    mask = dataset['displacement'] <= 0.99
    
    # Plot only the filtered data
    plt.plot(dataset['strain'][mask], dataset['stress'][mask], 
             label= name[i], marker='o', markersize=3)

plt.xlabel('Strain [-]')
plt.ylabel('Stress [MPa]')
plt.title('Stress-Strain Curves for All Tests (Displacement 1 mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



loading_times = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 100000, 1000000]
max_stress = [np.max(dataset['stress'][dataset['displacement'] <= 0.99]) 
                for dataset in data.values()]

# Fit a polynomial (you can change the degree)
log_times = np.log10(loading_times)
coeffs = np.polyfit(log_times, max_stress, 1)  # Linear fit in log space
fit_line = np.polyval(coeffs, log_times)

plt.figure(figsize=(8, 6))
plt.plot(loading_times, max_stress, 'o', markersize=8, label='Data')
plt.plot(loading_times, fit_line, '--', linewidth=2, label='Fit')
plt.xlabel('Loading Time [ms]')
plt.ylabel('Max Stress [MPa]')
plt.title('Maximum Stress vs Loading Time')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.legend()
plt.show()


loading_times = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 100000, 1000000]
max_stress = [np.max(dataset['stress'][dataset['displacement'] <= 0.99]) 
            for dataset in data.values()]

# Create smooth curve
log_times = np.log10(loading_times)
log_times_smooth = np.linspace(log_times[0], log_times[-1], 100)
times_smooth = 10**log_times_smooth

# Fit polynomial through points
coeffs = np.polyfit(log_times, max_stress, 3)  # 2nd degree polynomial
stress_smooth = np.polyval(coeffs, log_times_smooth)

print(f"\nFitted function:")
print(f"Max Stress = {coeffs[0]:.4f} * (log10(time))^2 + {coeffs[1]:.4f} * log10(time) + {coeffs[2]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(loading_times, max_stress, 'o', markersize=8, label='Data')
plt.plot(times_smooth, stress_smooth, '-', linewidth=2, label='Fit')
plt.xlabel('Loading Time [ms]')
plt.ylabel('Max Stress [MPa]')
plt.title('Maximum Stress vs Loading Time')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.legend()
plt.show()


