import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the directories and corresponding file names
radii = [0,1,2,3,4,5,6,7]
actual_radii = [2500,3000, 3500,4000,5000,6000,7000,8000]
base_dir = "sphere"
file_template = "s0_c1p1.dat"

# Initialize lists to store the arrays
times_arrays = []
value_arrays = []

for radius in radii:
    # Construct the directory and file path
    dir_name = f"{base_dir}{radius}"
    file_name = file_template.format(radius)
    file_path = os.path.join(dir_name, file_name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the data from the file
        data = np.loadtxt(file_path)
        
        # Separate the two columns
        column1 = data[:, 0]
        column2 = data[:, 1]
        
        # Store the columns in the lists
        times_arrays.append(column1)
        value_arrays.append(column2)
    else:
        print(f"File {file_path} does not exist.")

# Now `column1_arrays` and `column2_arrays` hold the data from each file
# For example, to access the first file's columns:
# times_arrays[0], value_arrays[0]



## Shift times so that they are somewhat aligned 
for i in range(len(radii)) :
	times_arrays[i] = times_arrays[i] - (actual_radii[i]-actual_radii[0])

#print(times_arrays[0],times_arrays[1],times_arrays[2])	

###Identify the latest time of the furdest array:
time_cutoff = 14000
time_cutoff2 = time_cutoff
filtered_times_arrays = []
filtered_value_arrays = []
for i in range(len(radii)) :
	filtered_times_arrays.append(times_arrays[i][times_arrays[i] <= time_cutoff])
	filtered_value_arrays.append(value_arrays[i][times_arrays[i] <= time_cutoff])			

# Create a new figure and axis for the plot
plt.figure(figsize=(10, 6))

#for i in range(len(radii)) :
    #filtered_value_arrays[i] = (filtered_value_arrays[i])*actual_radii[i]
    #filtered_value_arrays[i] = (filtered_value_arrays[i]- filtered_value_arrays[i][0])
# Loop through each component and plot the corresponding time vs value arrays
for i in range(len(filtered_times_arrays)):
    plt.plot(filtered_times_arrays[i], filtered_value_arrays[i], label=f'Dataset {i+1}')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Values vs Time for Different Datasets')

# Adding a legend to differentiate the datasets
plt.legend()

# Show the plot
plt.show()

###after inspcting visually find a lower time cuttof
time_cutoff = 11000
filtered_times_arrays_again = []
filtered_value_arrays_again = []
for i in range(len(radii)) :
	filtered_times_arrays_again.append(filtered_times_arrays[i][time_cutoff <=  filtered_times_arrays[i]])
	filtered_value_arrays_again.append(filtered_value_arrays[i][time_cutoff <=  filtered_times_arrays[i]])	


# Loop through each component and plot the corresponding time vs value arrays
for i in range(len(filtered_times_arrays_again)):
    plt.plot(filtered_times_arrays_again[i], filtered_value_arrays_again[i], label=f'Dataset {i+1}')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Values vs Time for Different Datasets')

# Adding a legend to differentiate the datasets
plt.legend()

# Show the plot
plt.show()

# Define the model function for the sine wave
def sine_model(t,D, A, B, C):
    return D + A * np.sin(B * t + C)


A_fit = np.zeros(len(radii))
B_fit = np.zeros(len(radii))
C_fit = np.zeros(len(radii))
D_fit = np.zeros(len(radii))

# Fit the data using curve_fit
initial_guess = [-0.0, 1e-8, 0.0025, 0]  # Initial guess for A, B, and C

plot_times = np.linspace(time_cutoff,time_cutoff2,1000)

for i in range(len(radii)) :
    params, params_covariance = curve_fit(sine_model, filtered_times_arrays_again[i], filtered_value_arrays_again[i], p0=initial_guess)
    # Extract the fitted parameters
    D_fit[i], A_fit[i], B_fit[i], C_fit[i] = params

    print(f"Fitted parameters:  D ={D_fit[i]}, A ={A_fit[i]}, B = {B_fit[i]}, C = {C_fit[i]}")
    # Plot the results
    plt.scatter(filtered_times_arrays_again[i], filtered_value_arrays_again[i], label='Data')
    plt.plot(plot_times, sine_model(plot_times, D_fit[i], A_fit[i], B_fit[i], C_fit[i]), label='Fitted function', color='red')


plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Sine Wave Fit')
plt.show()


for i in range(len(radii)) :
    plt.scatter(actual_radii[i], abs(A_fit[i]), label='Data')

plt.legend()
plt.xlabel('R')
plt.ylabel('A')
plt.title('Amplitude as function of radius')
plt.show()


# Define the model function for the R extrapolation
def extrapolation(R, B, C):
    R = np.asarray(R)
    return B + C/R

initial_guess = [1, 1]  # Initial guess for A, B, and C
params, params_covariance = curve_fit(extrapolation, actual_radii[-4:], abs(A_fit[-4:]), p0=initial_guess)
# Extract the fitted parameters
B_e_Fit, C_e_Fit = params

print(f"Fitted parameters: B = {B_e_Fit}, C = {C_e_Fit}")
print(actual_radii)
print(actual_radii[-4:])
plot_radii = np.linspace(actual_radii[0],actual_radii[-1],1000)


# Plot the results
plt.scatter(actual_radii, abs(A_fit), label='Data')
plt.plot(plot_radii, extrapolation(plot_radii, B_e_Fit, C_e_Fit), label='Fitted function', color='red')

plt.legend()
plt.xlabel('R')
plt.ylabel('B')
plt.title('Stuff')
plt.show()


print("The value extracted to infinity is =",B_e_Fit)

