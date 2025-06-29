import os
from dotenv import load_dotenv
from zowe.zos_files_for_zowe_sdk import Datasets,Files
from zowe.zos_jobs_for_zowe_sdk import Jobs
from zowe.core_for_zowe_sdk import ProfileManager

# Load environment variables
load_dotenv('./.env') 

profile_manager = ProfileManager().load(profile_name='zosmf')
files_api = Files(connection=profile_manager)

# Set environment variables for Mainframe host, port, user credentials
MAINFRAME_HOST = os.environ.get('MAINFRAME_HOST')
MAINFRAME_PORT = int(os.environ.get('MAINFRAME_PORT'))
ZOWE_USERNAME = os.environ.get('MAINFRAME_USERNAME')
ZOWE_PASSWORD = os.environ.get('MAINFRAME_PASSWORD')

profile = {
    "host": MAINFRAME_HOST,
    "port": MAINFRAME_PORT,
    "username": ZOWE_USERNAME,  
    "password": ZOWE_PASSWORD,
    "ssl_verification": True, 
    "rejectUnauthorized": False,
    "protocol": "https"
}

print(f"Connecting to Zowe API at {MAINFRAME_HOST}:{MAINFRAME_PORT} and user {ZOWE_USERNAME} password {ZOWE_PASSWORD} and {len(ZOWE_PASSWORD)=}")

# Initialize Jobs and Files services
jobs_api = Jobs(connection=profile)
# files_api = Datasets(connection=profile)
files1 = Files(connection=profile)


def get_job_log(job_name: str, job_id: str) -> str:
    """Get job log for a given job name and ID"""
    try:
        # Get job details
        job = jobs_api.get_job_status(jobname=job_name, jobid=job_id)
        if job:
            # Get job spool files
            spool_files = jobs_api.list_spool_files(jobname=job_name, jobid=job_id)
            # Combine all spool content
            log_content = ""
            for spool_file in spool_files:
                log_content += jobs_api.get_spool_content(
                    jobname=job_name,
                    jobid=job_id,
                    spoolfileid=spool_file['id']
                )
            return log_content
        return "Job not found"
    except Exception as e:
        raise Exception(f"Error retrieving job log: {str(e)}")

def download_dataset(dataset_name: str, local_file: str) -> None:
    """Download a dataset with validation"""
    try:
        # Check if dataset exists
        print(f"Checking if dataset {dataset_name} exists...")
        # attr = files_api.list(name_pattern=dataset_name,return_attributes=False)
        # print(f"Dataset attributes: {attr}")
        # if not files_api.dataset_exists(dataset_name=dataset_name):
            # raise FileNotFoundError(f"Dataset {dataset_name} does not exist")

        # Get dataset info
        # dataset_info = files_api.get_dataset_attributes(dataset_name=dataset_name)
        
        # Validate dataset organization (PS)
        # if dataset_info.get('dataSetOrganization') != 'PS':
            # raise ValueError("Only physical sequential datasets are supported")

        # Download the dataset
        # files_api.download(
        #     dataset_name=dataset_name,
        #     output_file=local_file
        # )
        print(f"{files_api=}")
        files_api.delete_data_set(dataset_name=dataset_name)
    except Exception as e:
        raise Exception(f"Error downloading dataset: {str(e)}")

def submit_job(dataset_name: str, member_name: str = None) -> str:
    """Submit a job and return the job ID"""
    try:
        if member_name:
            # Submit from PDS member
            response = jobs_api.submit_job_from_dataset(
                dataset_name=f"{dataset_name}({member_name})"
            )
        else:
            # Submit from sequential dataset
            response = jobs_api.submit_job_from_dataset(
                dataset_name=dataset_name
            )
        return response['jobid']
    except Exception as e:
        raise Exception(f"Error submitting job: {str(e)}")


if __name__ == "__main__":
    # Example usage
    dataset_name = "TEST.DATA"
    local_file = "local_copy.txt"

    # Download dataset  
    try:
        download_dataset(dataset_name, local_file)
        print(f"Dataset {dataset_name} downloaded to {local_file}")
    except Exception as e:
        print(f"Error: {str(e)}")