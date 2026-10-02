import boto3
import os
import nibabel as nib
import numpy as np
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LakeFSLoader():
    """
    For loading files from lakefs repo.
    
    Uses lazy initialization of the boto3 client so that each DataLoader
    worker process creates its own independent SSL connection.
    """
    def __init__(self, endpoint, repo_name, local_cache_path, ca_path):
        self.endpoint = endpoint
        self.repo_name = repo_name
        self.local_cache_path = local_cache_path
        self.ca_path = ca_path
        self._client = None

    def _get_client(self):
        """Lazily initialize the boto3 client (creates a fresh one per process)."""
        if self._client is None:
            self._client = boto3.client('s3', endpoint_url=self.endpoint, verify=self.ca_path)
        return self._client

    def get_file(self, object_name):
        local_path = os.path.join(self.local_cache_path, object_name)

        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self._get_client().download_file(self.repo_name, object_name, local_path)

        return local_path

    def load_volume(self, object_name):
        """
        Load a NIfTI volume file and return as numpy array.
        
        Args:
            object_name: Path to the NIfTI file in the lakefs repository
            
        Returns:
            numpy array: Volume data with shape (H, W, D)
        """
        local_path = self.get_file(object_name)
        nii = nib.load(local_path)
        volume = nii.get_fdata()
        return np.array(volume)
