import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.optimize import curve_fit

def get_program_directory():
    script_path = os.path.abspath(__file__)
    program_directory = os.path.dirname(script_path)
    return program_directory

# Define the function to fit
def funcbk(f, b, k):
    return (f/b)**k

# Get the program directory
program_dir = get_program_directory()

# Get a list of all txt and csv files in the program directory
txt_files = sorted(glob.glob(os.path.join(program_dir, '*fitparams.txt')))
csv_files = sorted(glob.glob(os.path.join(program_dir, '*fitpoints.csv')))

# Make sure we have the same number of txt and csv files
assert len(txt_files) == len(csv_files)

# Define the y position for the text box, starting from the top
y_pos = 0.98

# Define a list of colors for the plots and text boxes
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# dT/dt values for H = 1000, 2000, 3000 A/m
dTdt_values = np.empty((0,3), int)

plt.figure(figsize=(11, 9))
# Process each pair of files
for i, (txt_file, csv_file) in enumerate(zip(txt_files, csv_files)):
    # Read the parameters from the txt file
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        a = float(lines[1].split('=')[1].split('±')[0].split()[0])
        n = float(lines[2].split('=')[1].split('±')[0].split()[0])

    # Define the function for the curve
    def func(H):
        return (H/a)**n

    dTdt_values = np.vstack([dTdt_values, [(1000/a)**n, (2000/a)**n, (3000/a)**n]])
    
    # Read the data points from the csv file
    data = pd.read_csv(csv_file)

    # Create a range of H values for the curve
    H_values = np.linspace(data['H'].min(), data['H'].max(), 100)

    # Plot the curve with a specific color
    plt.plot(H_values, func(H_values), label=f'Fit: $(H/{a})^{n}$', color=colors[i % len(colors)])

    # Plot the data points with the same color
    plt.scatter(data['H'], data['dT/dt'], label=f'Data points from {csv_file}', color=colors[i % len(colors)])

    # Extract the number from the file name
    number = os.path.basename(txt_file).split('fitparams.txt')[0]

    # Add the contents of the txt file as a text box on the plot with the same color
    with open(txt_file, 'r') as f:
        txt_content = f.read()
    plt.gca().text(0.05, y_pos, number + " [kHz]\n" + txt_content, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), color=colors[i % len(colors)])

    # Adjust the y position for the next text box
    y_pos -= 0.2

    
# Set labels, title and legend
plt.xlabel('H [A/m]')
plt.ylabel('dT/dt [mK/s]')
plt.title('EFH-1, RMF')
plt.xlim(0)  # Set x-axis to start at 0
plt.ylim(0)  # Set y-axis to start at 0
plt.savefig(os.path.join(program_dir, 'dopasowanie.jpeg'), format='jpeg')
plt.clf()

# Frequencies in kHz
frequencies = np.array([50, 100, 200])

print(dTdt_values)

# Initial guess for the parameters
initial_guess = [1000, 2]

# Fit the function to each set of dT/dt values and plot the results
for j, dTdt in enumerate(dTdt_values):
    # Fit the function to the data
    popt, pcov = curve_fit(funcbk, frequencies, dTdt, p0=initial_guess)

    print("Fitted parameters for H =", 1000*(j+1), "A/m: b =", popt[0], ", k =", popt[1])

    # Plot the data and the fitted curve
    plt.plot(frequencies, dTdt, 'o', color=colors[(i+j) % len(colors)], label='Data for H = '+str(1000*(j+1))+' A/m')
    # Generate a range of frequencies for the fitted curve
    fitted_frequencies = np.linspace(0, 200, 400)
    plt.plot(fitted_frequencies, funcbk(fitted_frequencies, *popt), '-', color=colors[(i+j) % len(colors)], label='Fit for H = '+str(1000*(j+1))+' A/m: b=%5.1f, k=%5.2f' % tuple(popt))

plt.title('EFH-1, RMF')
plt.xlabel('f [kHz]')
plt.ylabel('dT/dt [mK/s]')
plt.xlim(0)  # Set x-axis to start at 0
plt.ylim(0)  # Set y-axis to start at 0
plt.legend()
plt.savefig(os.path.join(program_dir, 'dopasowanie_bk.jpeg'), format='jpeg')
plt.clf()

#SAR calculions
#specific heat capacity at static pressure
Cp = 1840

# Calculate the SAR values
data['SAR'] = Cp * data['dT/dt']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate through the CSV files and plot the extrapolated curves
for csv_file in csv_files:
    data = pd.read_csv(csv_file)
    Cp = 1840/1000#J/gK
    data['SAR'] = Cp * data['dT/dt']/1000#mK

    # Fit a power function to the data
    def power_func(x, a, b):
        return (x/a)**b

    popt, pcov = curve_fit(power_func, data['H'], data['SAR'])
    a, b = popt

    # Extract the number from the file name
    number = os.path.basename(csv_file).split('fitpoints.csv')[0]

    # Extrapolate the fitted curve to H = 10000
    h_extrapolate = np.linspace(0, 10000, 1000)
    sar_extrapolate = power_func(h_extrapolate, a, b)
    ax.plot(h_extrapolate, sar_extrapolate, label=f'{number} [kHz]')
    print(f'Extrapolated SAR at H = 10000 for Sample {number}: {sar_extrapolate[-1]:.2f}')
    
    target_sar = 0.1
    idx = next((i for i, x in enumerate(sar_extrapolate) if x >= target_sar), None)

    if idx is not None:
        # Mark the point on the plot
        plt.plot(h_extrapolate[idx], target_sar, 'ro', markersize=5)
    
        # Add the x-value next to the marked point
        plt.text(h_extrapolate[idx], target_sar, f'{h_extrapolate[idx]:.0f} A/m  ', va='bottom', ha='right', fontsize=10)

# Add labels and title
ax.axhline(y=0.1, color='r', linestyle='--', label='effective heating')
ax.set_xlabel('H [A/m]')
ax.set_ylabel('SAR [W/g]')
ax.set_title('Extrapolated SAR(H) up to 10 [kA/m]')
ax.grid()
ax.legend()
plt.savefig(os.path.join(program_dir, 'SAR.jpeg'), format='jpeg')
