import subprocess
import time
import csv

# Define the list of datasets and methods
DATANAMES = ['acs-income']  # adult
METHODS = ['tabsyn', 'tabddpm','stasy']
N_ITERATIONS = 3

timing_baselines = 'timing_baselines.csv'

# Header for the CSV file
header = ['dataset', 'iteration','method', 'stage', 'time']

with open(timing_baselines, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
for dataname in DATANAMES:
    # Preprocess data for the baselines
    print(f'Preprocess data for {dataname}')
    subprocess.run(["python", "process_dataset.py", "--dataname", dataname], capture_output=True, text=True)
    for i in range(N_ITERATIONS):
        for method in METHODS:        
            # Train the model
            print(f'Train {method} for {dataname}')
            tt0 = time.time()
            if method=='tabsyn':
                subprocess.run(
                    ["python", "main.py", "--dataname", dataname, "--method", "vae", "--mode", "train"],
                    capture_output=False, text=True
                )
                tt_vae = time.time() - tt0
                with open(timing_baselines, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataname,i, 'vae_part', "train", tt_vae])
                subprocess.run(
                    ["python", "main.py", "--dataname", dataname, "--method", "tabsyn", "--mode", "train"],
                    capture_output=False, text=True
                )
                tt_tabsyn = tt_vae - tt0
                with open(timing_baselines, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataname, i, 'tabsyn_part', "train", tt_tabsyn])
                
            else:    
                subprocess.run(
                    ["python", "main.py", "--dataname", dataname, "--method", method, "--mode", "train"],
                   capture_output=False, text=True
                )
            tt = time.time() - tt0
            with open(timing_baselines, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([dataname, i, method, "train", tt])

            # Generate (sampling)
            gt0 = time.time()
            subprocess.run(
                ["python", "main.py", "--dataname", dataname, "--method", method, "--mode", "sample"],
                capture_output=False, text=True
            )
            gt = time.time() - gt0
            with open(timing_baselines, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([dataname, i, method, "sample", gt])


