from datetime import datetime
import os
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

from node import Node

SESSION_ID = datetime.now().strftime("%Y%m%d")

def add_row(folder_name, alpha):
    db_file = f"{folder_name}/log.db"

    # Connect to the database
    db = sqlite3.connect(db_file)
    cur = db.cursor()

    # Execute a query to retrieve the time that batteries are battery greater then zero
    cur.execute("""
            SELECT 
            SUM(time)
        FROM end_batteries
    """)
    
    ts = cur.fetchone()

    # Execute a query to retrieve the required aggregated values from the round table
    cur.execute("""
        SELECT 
            AVG(variance)
        FROM round
        WHERE time > 3500
    """)

    # Fetch the aggregated values from the result
    round_data = cur.fetchone()
    
    # Max time, Min time, Max battey when first dies
    cur.execute("""
        SELECT 
            MIN(time),
            MAX(time)
        FROM end_batteries
    """)

    # Fetch the aggregated values from the result
    lifespans = cur.fetchone()
    
    cur.execute("""
                SELECT max_battery
                FROM end_batteries
                WHERE time = (SELECT MIN(time) FROM end_batteries)
            """)

    divergent = cur.fetchone()
    

    res = cur.execute( f'''
        select 
            cast(finish_time as int), count(*)
        from (
            select 
                id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
            from 
                jobs where node_uid = 0 and type = 0 and executed = 1 and rejected = 0 and generated_at > 3500
            ) 
        where lag_time > 0
        group by cast(finish_time as int)
    ''')
    #count all lines
    jobs_0 = res.fetchall()
    n_jobs_0 = jobs_0.__len__()
    
    res = cur.execute( f'''
        select 
            cast(finish_time as int), count(*)
        from (
            select 
                id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
            from 
                jobs where node_uid = 0 and type = 1 and executed = 1 and rejected = 0 and generated_at > 3500
            ) 
        where lag_time > 0
        group by cast(finish_time as int)
    ''')
    jobs_1 = res.fetchall()
    n_jobs_1 = jobs_1.__len__()
    
    res = cur.execute( f'''
        select 
            cast(finish_time as int), count(*)
        from (
            select 
                id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
            from 
                jobs where node_uid = 0 and type = 1 and executed = 1 and rejected = 0 and generated_at > 3500
            ) 
        where lag_time > 0
        group by cast(finish_time as int)
    ''')
    jobs_2 = res.fetchall()
    n_jobs_2 = jobs_2.__len__()

    jobs_data_0 = sum(1 for job in jobs_0 if job[1] >= 50)
    jobs_data_1 = sum(1 for job in jobs_1 if job[1] >= 20)
    jobs_data_2 = sum(1 for job in jobs_2 if job[1] >= 10)

    # Close the cursor and database connection
    cur.close()
    db.close()

    # Combine the aggregated values from both tables into a single row
    row = [
    f"{alpha}",  # Alpha value
    f"{round(round_data[0], 2):.3f}",  # Average variance rounded to 2 decimal places
    f"{round(divergent[0], 2):.2f}",  # Difference of max and min battery rounded to 2 decimal places
    f"{int(round(lifespans[0]))}",    # Minimum lifespan as an integer
    f"{int(round(lifespans[1]))}",    # Maximum lifespan as an integer
    f"{round(((jobs_data_0 + jobs_data_1 + jobs_data_2) / (n_jobs_0 + n_jobs_1 + n_jobs_2)) * 100, 2):.1f}%",  # Deadline ratio rounded to 2 decimal places and converted to percentage
    f"{round(jobs_data_0 / n_jobs_0 * 100, 2):.1f}%",  # Deadline ratio for job type 0 rounded to 2 decimal places and converted to percentage
    f"{round(jobs_data_1/ n_jobs_1 * 100, 2):.1f}%",  # Deadline ratio for job type 1 rounded to 2 decimal places and converted to percentage
    f"{round(jobs_data_2 / n_jobs_2 * 100, 2):.1f}%"   # Deadline ratio for job type 2 rounded to 2 decimal places and converted to percentage
    f"{ts[0]}"
    ]
    
    rows.append(row)
        
# Define the base folder containing the database folders
base_folder = "_log/learning/D_SARSA/ONLY_WORKERS"

# Define the range of alpha values
alpha_range = [i / 100 for i in range(0, 101, 5)]  # [0.0, 0.5, 1.0]

# Create an empty list to store the rows
rows = []
        
# Iterate over each alpha value
for alpha in alpha_range:
    # Set the folder name based on the alpha value
    folder_name = f"{os.getcwd()}/{base_folder}/{SESSION_ID}_{alpha:.2f}"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        add_row(folder_name, f'{alpha:.2f}')
        
# Define the base folder containing the database folders
base_folder = "_log/no-learning"

# Define the range of alpha values
policies =  [Node.NoLearningPolicy.LEAST_LOADED_NOT_AWARE,Node.NoLearningPolicy.MAXIMUM_LIFESPANE,Node.NoLearningPolicy.RANDOM]

for policy in policies:
    # Set the folder name based on the alpha value
    folder_name = f"{os.getcwd()}/{base_folder}/{policy.name}/{SESSION_ID}_{policy.name}_ONLY_WORKERS"
    print(folder_name)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Get the database file path
        add_row(folder_name, policy.name)
       

file_path = "table_data_ONLY_WORKERS.txt"

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