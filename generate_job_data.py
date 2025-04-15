import pandas as pd
import random
import os
import duckdb  # Add this import

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

    # Randomly assign criticality (Y/N)
    is_critical = 'Y' if random.random() < 0.3 else 'N'
    
    data.append({
        'jobname': job,
        'predecessors': predecessors,
        'successors': successors,
        'file_triggers': file_trigger,
        'is_critical': is_critical
    })

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Create DataFrame and save to CSV in data directory
df = pd.DataFrame(data)
output_file = os.path.join(data_dir, 'job_dependencies.csv')
df.to_csv(output_file, index=False)

print(f"Generated job dependencies file at: {output_file}")

# After creating the CSV file, add DuckDB logic:
# Create database directory if it doesn't exist
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)

# Connect to DuckDB
db_path = os.path.join(db_dir, 'jobs_database.db')
conn = duckdb.connect(db_path)

try:
    # Drop the table first if it exists, then create it
    conn.execute("DROP TABLE IF EXISTS mainframe_deps")
    conn.execute("""
        CREATE TABLE mainframe_deps (
            job_name VARCHAR,
            predecessor_job VARCHAR,
            successor_job VARCHAR,
            file_trigger VARCHAR,
            is_critical VARCHAR(1)
        )
    """)
    
    # Transform the data for DuckDB
    db_records = []
    for record in data:
        job_name = record['jobname']
        predecessors = record['predecessors'].split(',') if record['predecessors'] else []
        successors = record['successors'].split(',') if record['successors'] else []
        
        # Add a record for each predecessor
        for pred in predecessors:
            if pred:  # Only add if predecessor exists
                db_records.append({
                    'job_name': job_name,
                    'predecessor_job': pred,
                    'successor_job': None,
                    'file_trigger': record['file_triggers'],
                    'is_critical': record['is_critical']
                })
        
        # Add a record for each successor
        for succ in successors:
            if succ:  # Only add if successor exists
                db_records.append({
                    'job_name': job_name,
                    'predecessor_job': None,
                    'successor_job': succ,
                    'file_trigger': record['file_triggers'],
                    'is_critical': record['is_critical']
                })
        
        # If no dependencies, still add the job with its file trigger
        if not predecessors and not successors:
            db_records.append({
                'job_name': job_name,
                'predecessor_job': None,
                'successor_job': None,
                'file_trigger': record['file_triggers'],
                'is_critical': record['is_critical']
            })

    # Convert to DataFrame and insert into DuckDB
    db_df = pd.DataFrame(db_records)
    conn.execute("DELETE FROM mainframe_deps")  # Clear existing data
    conn.execute("INSERT INTO mainframe_deps SELECT * FROM db_df")
    
    # Fetch and print all data from the table
    result = conn.execute("SELECT * FROM mainframe_deps where is_critical ='Y' ").fetchdf()
    print("\nData in mainframe_deps table:")
    print(result.to_string())
    
    print(f"\nData successfully saved to DuckDB at: {db_path}")

finally:
    conn.close()