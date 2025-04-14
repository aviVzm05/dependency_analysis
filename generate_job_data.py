import pandas as pd
import random
import os

# Create lists of job names
job_names = [f'JOB{str(i).zfill(4)}' for i in range(1, 101)]
all_jobs = ['JOB1001', 'JOB1002', 'JOB1003', 'JOB1004', 'JOB1005', 'JOB1006', 'JOB1007', 'JOB1008']

# Create file trigger patterns
file_patterns = [
    '/data/input/*.dat',
    '/logs/*.txt',
    '/feeds/daily/*.csv',
    '/batch/files/*.xml',
    '/imports/*.json',
    None
]

# Generate data
data = []
for job in job_names:
    # Random number of predecessors and successors (0-3)
    num_pred = random.randint(0, 3)
    num_succ = random.randint(0, 3)
    
    # Random selection of predecessors and successors
    predecessors = ','.join(random.sample(all_jobs, num_pred))
    successors = ','.join(random.sample(all_jobs, num_succ))
    
    # Random file trigger (70% chance of having one)
    file_trigger = random.choice(file_patterns) if random.random() < 0.7 else None
    
    data.append({
        'jobname': job,
        'predecessors': predecessors,
        'successors': successors,
        'file_triggers': file_trigger
    })

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Create DataFrame and save to CSV in data directory
df = pd.DataFrame(data)
output_file = os.path.join(data_dir, 'job_dependencies.csv')
df.to_csv(output_file, index=False)

print(f"Generated job dependencies file at: {output_file}")