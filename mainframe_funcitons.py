import os
from zowe.core.connection import ApiConnection
from zowe.apis.jobs_api import JobsApi
from zowe.apis.files_api import FilesApi

# Set environment variables for Mainframe host, port, user credentials
MAINFRAME_HOST = os.environ.get('MAINFRAME_HOST')
MAINFRAME_PORT = int(os.environ.get('MAINFRAME_PORT'))
ZOWE_USERNAME = os.environ.get('ZOWE_USERNAME')
ZOWE_PASSWORD = os.environ.get('ZOWE_PASSWORD')

# Initialize Zowe connection
connection = ApiConnection(
    host=MAINFRAME_HOST,
    port=MAINFRAME_PORT,
    user=ZOWE_USERNAME,
    password=ZOWE_PASSWORD
)

# Initialize Jobs and Files services
jobs_api = JobsApi(connection)
files_api = FilesApi(connection)

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
        if not files_api.dataset_exists(dataset_name=dataset_name):
            raise FileNotFoundError(f"Dataset {dataset_name} does not exist")

        # Get dataset info
        dataset_info = files_api.get_dataset_attributes(dataset_name=dataset_name)
        
        # Validate dataset organization (PS)
        if dataset_info.get('dataSetOrganization') != 'PS':
            raise ValueError("Only physical sequential datasets are supported")

        # Download the dataset
        files_api.download_dataset(
            dataset_name=dataset_name,
            output_file=local_file
        )
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