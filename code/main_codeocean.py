#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import multiprocessing
import os
from datetime import datetime
from plot_stacked_performances import plot_stacked
from plot_table_cloud import append_row
from run_simulations_no_learning_cloud import run_simulation as run_simulation_no_learning_cloud
from run_simulation_d_sarsa_cloud import run_simulation as run_simulation_cloud
from run_simulations_no_learning import run_simulation as run_simulation_no_learning
from run_simulation_d_sarsa import run_simulation as run_simulation
from run_simulation_d_sarsa_cloud_failure import run_simulation as run_simulation_failure
from node import Node

ALPHA_INCREMENT = 0.05
SESSION_ID = datetime.now().strftime("%Y%m%d")
PATH_RESULTS = "../results"
PATH_RESULTS_DATA = f"{PATH_RESULTS}/data"
PATH_RESULTS_TABLE = f"{PATH_RESULTS}/table"
PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"
os.makedirs(PATH_RESULTS_DATA, exist_ok=True)
os.makedirs(PATH_RESULTS_PLOT, exist_ok=True)
os.makedirs(PATH_RESULTS_TABLE, exist_ok=True)


# Calculate the number of processes to launch based on CPU cores
num_cores = multiprocessing.cpu_count()

# Calculate the range of alphas to cover
alpha_values =[i * ALPHA_INCREMENT for i in range(0, 21)]

# Launch processes
processes = []
for alpha in alpha_values:
    process = multiprocessing.Process(target=run_simulation, args=(alpha,))
    processes.append(process)
    process.start()

    # Control the number of running processes
    if len(processes) >= num_cores:
        for p in processes:
            p.join()
        processes = []

# Wait for remaining processes to finish
for p in processes:
    p.join()

print("All simulations completed (d-sarsa).")


policies =  [Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD, Node.NoLearningPolicy.MAXIMUM_LIFESPANE, Node.NoLearningPolicy.RANDOM]
# Launch processes
processes = []
for policy in policies:
    process = multiprocessing.Process(target=run_simulation_no_learning, args=(policy,))
    processes.append(process)
    process.start()

    # Control the number of running processes
    if len(processes) >= num_cores:
        for p in processes:
            p.join()
        processes = []

# Wait for remaining processes to finish
for p in processes:
    p.join()

print("All simulations completed. (no-learning)")

processes = []
for alpha in alpha_values:
    process = multiprocessing.Process(target=run_simulation_cloud, args=(alpha,))
    processes.append(process)
    process.start()

    # Control the number of running processes
    if len(processes) >= num_cores:
        for p in processes:
            p.join()
        processes = []

# Wait for remaining processes to finish
for p in processes:
    p.join()

print("All simulations completed (d-sarsa-cloud).")


policies =  [Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD, Node.NoLearningPolicy.MAXIMUM_LIFESPANE, Node.NoLearningPolicy.RANDOM]
# Launch processes
processes = []
for policy in policies:
    process = multiprocessing.Process(target=run_simulation_no_learning_cloud, args=(policy,))
    processes.append(process)
    process.start()

    # Control the number of running processes
    if len(processes) >= num_cores:
        for p in processes:
            p.join()
        processes = []

# Wait for remaining processes to finish
for p in processes:
    p.join()

print("All simulations completed. (no-learning-cloud)")

# Calculate the range of alphas to cover
alpha_values = [1.0, 0.0]

# Launch processes
processes = []
for alpha in alpha_values:
    process = multiprocessing.Process(target=run_simulation_failure, args=(alpha,))
    processes.append(process)
    process.start()

    # Control the number of running processes
    if len(processes) >= num_cores:
        for p in processes:
            p.join()
        processes = []

# Wait for remaining processes to finish
for p in processes:
    p.join()

print("All simulations completed (Failure).")

# Define the base folder containing the database folders
base_folder = f"{PATH_RESULTS_DATA}/_log/learning/D_SARSA/WORKERS_OR_CLOUD"

# Define the range of alpha values
alpha_range = [i / 100 for i in range(0, 101, 5)]  # [0.0, 0.5, 1.0]

# Create an empty list to store the rows
rows = []
        
# Iterate over each alpha value
for alpha in alpha_range:
    # Set the folder name based on the alpha value
    folder_name = f"{base_folder}/{SESSION_ID}_{alpha:.2f}"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        rows.append(append_row(folder_name, f'{alpha:.2f}'))
        
# Define the base folder containing the database folders
base_folder = f"{PATH_RESULTS_DATA}/_log/no-learning"

# Define the range of alpha values
policies =  [Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,Node.NoLearningPolicy.MAXIMUM_LIFESPANE,Node.NoLearningPolicy.RANDOM]

for policy in policies:
    # Set the folder name based on the alpha value
    folder_name = f"{base_folder}/{policy.name}/{SESSION_ID}_{policy.name}_WORKERS_OR_CLOUD"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        rows.append(append_row(folder_name, policy.name))
       


os.makedirs(f"{PATH_RESULTS_TABLE}/", exist_ok=True)
file_path = f"{PATH_RESULTS_TABLE}/table_data_WORKERS_OR_CLOUD.txt"

# Define table data with headers
table_data = [
    ["Alpha","Mean Difference Batteries", "Max - Min batteries when first node die", "Min Lifespan", "Max Lifespan", "deadline / total job", "deadline / total type_0", "deadline / total type_1", "deadline / total type_2", "Time Service"]
]

# Assuming 'rows' contains the data rows to be added to the table
for row in rows:
    table_data.append(row)

# Open the file in write mode
with open(file_path, "w") as file:
    # Write each row of the table data to the file
    file.write
    for row in table_data:
        # Join the elements of the row with a tab delimiter and write to the file
        file.write("\t".join(map(str, row)) + "\n")

# Notify that the file has been created
print(f"Table data has been saved to '{file_path}'.")

# Define the base folder containing the database folders
base_folder = f"{PATH_RESULTS_DATA}/_log/learning/D_SARSA/ONLY_WORKERS"

# Define the range of alpha values
alpha_range = [i / 100 for i in range(0, 101, 5)]  # [0.0, 0.5, 1.0]

# Create an empty list to store the rows
rows = []
        
# Iterate over each alpha value
for alpha in alpha_range:
    # Set the folder name based on the alpha value
    folder_name = f"{base_folder}/{SESSION_ID}_{alpha:.2f}"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        rows.append(append_row(folder_name, f'{alpha:.2f}'))
        
# Define the base folder containing the database folders
base_folder = f"{PATH_RESULTS_DATA}/_log/no-learning"

# Define the range of alpha values
policies =  [Node.NoLearningPolicy.LEAST_LOADED_NOT_AWARE,Node.NoLearningPolicy.MAXIMUM_LIFESPANE,Node.NoLearningPolicy.RANDOM]

for policy in policies:
    # Set the folder name based on the alpha value
    folder_name = f"{base_folder}/{policy.name}/{SESSION_ID}_{policy.name}_ONLY_WORKERS"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        rows.append(append_row(folder_name, policy.name))
       


os.makedirs(f"{PATH_RESULTS_TABLE}/", exist_ok=True)
file_path = f"{PATH_RESULTS_TABLE}/table_data_ONLY_WORKERS.txt"

# Define table data with headers
table_data = [
    ["Alpha","Mean Difference Batteries", "Max - Min batteries when first node die", "Min Lifespan", "Max Lifespan", "deadline / total job", "deadline / total type_0", "deadline / total type_1", "deadline / total type_2", "Time Service"]
]

# Assuming 'rows' contains the data rows to be added to the table
for row in rows:
    table_data.append(row)

# Open the file in write mode
with open(file_path, "w") as file:
    # Write each row of the table data to the file
    file.write
    for row in table_data:
        # Join the elements of the row with a tab delimiter and write to the file
        file.write("\t".join(map(str, row)) + "\n")

# Notify that the file has been created
print(f"Table data has been saved to '{file_path}'.")

alpha_values = ["0.00", "1.00"]
type = "_FAILURE"

for alpha in alpha_values:
    plot_stacked(SESSION_ID, alpha, type, PATH_RESULTS_PLOT, PATH_RESULTS_DATA)


