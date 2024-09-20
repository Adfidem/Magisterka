import glob
import os
import numpy as np
from scipy.stats import linregress
import configparser
import re
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_settings(folder_path):
    settings = configparser.ConfigParser()
    settings_file_path = os.path.join(folder_path, 'config', 'settings.txt')

    # Check if settings file exists
    if os.path.isfile(settings_file_path):
        settings.read(settings_file_path)
    else:
        # Create new settings file
        settings['DEFAULT'] = {'LineFitTimeStart': 40, 'LineFitTimeEnd': 60, 'outputCSVcolumnXname': 'H', 'outputCSVcolumnYname': 'dT/dt', 'XcollumnMultiplier': 1, 'YcollumnMultiplier': 1000, 'FittedPlotXaxis': 't [s]', 'FittedPlotYaxis': 'T [K]', 'frequency': 'NA [kHz]'}
        os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)
        with open(settings_file_path, 'w') as config_file:
            settings.write(config_file)

    return settings

class DataObject:
    def __init__(self, file_name, name):
        self.file_name = file_name
        self.name = name
        self.data = np.loadtxt(file_name, skiprows=1, delimiter='\t', encoding='utf-16le')

def load_data_objects(folder_path):
    file_paths = glob.glob(folder_path + '/*.txt')

    data_objects = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        data_object = DataObject(file_path, file_name)
        data_objects.append(data_object)

    return data_objects

def fit_linear_function(data_object, LineFitTimeStart, LineFitTimeEnd):
    x_values = []
    y_values = []

    # Extract relevant data points
    for x, y in data_object.data:
        x = float(x)
        y = float(y)

        if LineFitTimeStart <= x <= LineFitTimeEnd:
            x_values.append(x)
            y_values.append(y)

    if not x_values or not y_values:
        return None

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

    return slope, intercept, r_value, p_value, std_err

def get_program_directory():
    script_path = os.path.abspath(__file__)
    program_directory = os.path.dirname(script_path)
    return program_directory

def extract_number(string):
    # Use regular expression to find the number
    match = re.search(r'\d+', string)

    if match:
        number = int(match.group())
        return number
    else:
        # No number found in the string
        return "NAN"

def save_data_to_txt(data_points, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as file:
        for point in data_points:
            file.write(f'{point[0]}\t{point[1]}\n')

def save_data_to_csv(data_points, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([settings.get('DEFAULT', 'outputCSVcolumnXname'), settings.get('DEFAULT', 'outputCSVcolumnYname')])
        writer.writerows(data_points)

def plot_data_with_linear_function(data_object, slope, intercept, output_file):
    x_values = data_object.data[:, 0].astype(float)
    y_values = data_object.data[:, 1].astype(float)
    
    # Create a new figure for each plot
    plt.figure()
    
    # Plot data points
    plt.scatter(x_values, y_values, s=10, label='Data Points')

    # Generate x values for the linear function line
    x_line = np.linspace(min(x_values), max(x_values), 100)

    # Calculate y values for the linear function line
    y_line = slope * x_line + intercept

    # Plot the linear function line
    plt.plot(x_line, y_line, 'r', label='Linear Function')
    label = data_object.name.replace('Am', '[A/m]')
    # Set plot title and labels
    plt.title(label)
    plt.xlabel(settings.get('DEFAULT', 'FittedPlotXaxis'))
    plt.ylabel(settings.get('DEFAULT', 'FittedPlotYaxis'))

    # Show legend
    plt.legend()

    # Save the plot as a JPEG file
    plt.savefig(output_file, format='jpeg')

    # Close the plot
    plt.close()

# Get the program directory
folder_path = get_program_directory()

# Load the settings
settings = load_settings(folder_path)

# Load data objects from the folder
data_objects = load_data_objects(folder_path)

# Create a list to store the output data
OutputData = [(0, 0)]

if not os.path.exists(os.path.join(folder_path, 'output')):
    os.makedirs(os.path.join(folder_path, 'output'))

# Iterate over data objects, perform calculations and create plots
for data_object in data_objects:
    result = fit_linear_function(data_object, settings.getfloat('DEFAULT', 'LineFitTimeStart'), settings.getfloat('DEFAULT', 'LineFitTimeEnd'))

    if result is None:
        outPoint = ("No valid data points found within the specified range for", data_object.name)
        OutputData.append(outPoint)
    else:
        slope, intercept, r_value, p_value, std_err = result

        output_file = os.path.join(folder_path, 'output', f'{data_object.name}.jpeg')

        plot_data_with_linear_function(data_object, slope, intercept, output_file)
        outPoint = (extract_number(data_object.name)*settings.getfloat('DEFAULT', 'XcollumnMultiplier'),slope*settings.getfloat('DEFAULT', 'YcollumnMultiplier'))
        OutputData.append(outPoint)


# Sort the output data based on the second column
OutputData = sorted(OutputData, key=lambda x: x[1])

# Define the output file path
output_file_path = os.path.join(folder_path, 'output')

# Save data to txt file
txt_filename = str(extract_number(settings.get('DEFAULT','frequency'))) + 'fitpoints.txt'
save_data_to_txt(OutputData, output_file_path, txt_filename)

# Save data to csv file
csv_filename = str(extract_number(settings.get('DEFAULT','frequency'))) + 'fitpoints.csv'
save_data_to_csv(OutputData, output_file_path, csv_filename)

H, dT_dt = zip(*OutputData)

# Convert the lists to numpy arrays
H = np.array(H)
dT_dt = np.array(dT_dt)

# Define the function to fit
def func(H, a, n):
    return (H / a) ** n

# Use curve_fit to find optimal parameters
popt, pcov = curve_fit(func, H, dT_dt)

# Calculate the standard deviation of the parameters
perr = np.sqrt(np.diag(pcov))

# Create a range of x values
H_values = np.linspace(min(H), max(H), 500)

# Calculate the corresponding y values
dT_dt_values = func(H_values, *popt)

# Calculate the residuals
residuals = dT_dt - func(H, *popt)

# Calculate the total sum of squares
ss_total = np.sum((dT_dt - np.mean(dT_dt)) ** 2)

# Calculate the residual sum of squares
ss_res = np.sum(residuals ** 2)

# Calculate the R-squared value
r_squared = 1 - (ss_res / ss_total)

# Calculate the adjusted R-squared value
n = len(H)  # number of data points
p = len(popt)  # number of parameters
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Create the fitting plot
plt.figure(figsize=(10, 6))
plt.plot(H, dT_dt, 'bo', label='Data points')
plt.plot(H_values, dT_dt_values, 'r-', label='Fit')

# Create a text box with the fit parameters, the adjusted R-squared value, and the equation
textstr = '\n'.join((
    r'$\frac{dT}{dt} = \left(\frac{H}{%.0f}\right)^{%.2f}$' % (popt[0], popt[1]),
    r'$\mathrm{a}=%.0f \pm %.0f$' % (popt[0], perr[0]),
    r'$\mathrm{n}=%.2f \pm %.2f$' % (popt[1], perr[1]),
    r'$\mathrm{adj.} \ R^2=%.4f$' % adjusted_r_squared))
print(textstr)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.1)

# place a text box in upper left in axes coords
plt.gca().text(0.015, 0.85, textstr, transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.title('EFH-1, RMF, ' + settings.get('DEFAULT','frequency'))
plt.xlabel('H [A/m]')
plt.ylabel('dT/dt [mK/s]')
plt.legend()
plt.savefig(os.path.join(folder_path, 'output', 'fitting.jpeg'), format='jpeg')

# Save data to txt file
txt_filename = str(extract_number(settings.get('DEFAULT','frequency'))) + 'fitparams.txt'
with open(os.path.join(output_file_path, txt_filename), 'w') as file:
        file.write(textstr)

# Save data to csv file
csv_filename = str(extract_number(settings.get('DEFAULT','frequency'))) + 'fitparams.csv'
with open(os.path.join(output_file_path, csv_filename), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([textstr])

# Create the temperature plot
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'gray']  # blue, green, red, cyan, magenta, yellow, black

# Find the global minimum y-value across all data_objects
global_min_y = min(min(data_object.data[30:, 1].astype(float)) for data_object in data_objects)

# Create a list of tuples with the data_object and its maximum normalized value
data_object_max_values = [(data_object, max(data_object.data[30:, 1].astype(float) - data_object.data[30, 1].astype(float) + global_min_y)) for data_object in data_objects]

# Sort the list of tuples based on the maximum normalized value in descending order
data_object_max_values.sort(key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
for i, (data_object, _) in enumerate(data_object_max_values):
    x_values = data_object.data[30:, 0].astype(float)
    y_values = data_object.data[30:, 1].astype(float)
    x_values -= 30
    y_values -= y_values[0]  # shift y-values so the minimum is at zero
    y_values += global_min_y
    label = data_object.name.replace('Am', '[A/m]')
    plt.plot(x_values, y_values, color=colors[i % len(colors)], label=label)
plt.title('EFH-1, RMF, ' + settings.get('DEFAULT','frequency'), fontsize=16)
plt.xlabel('t [s]', fontsize=14)
degree_symbol = '\u00B0'
plt.ylabel(f'T [{degree_symbol}C]', fontsize=14)
plt.legend(fontsize=14)
plt.savefig(os.path.join(folder_path, 'output', 'temp.jpeg'), format='jpeg')
plt.show()


