import os
import logging
import requests
from .config import RAW_INPUT_DIR, CLEAN_DATA_DIR

# Set up logging
logger = logging.getLogger(__name__)

# Likely files needed based on preprocessing.py
# Schema: bacteria.{seq}.{tissue}.{level}.{ext}.txt
# and metadata.{seq}.{tissue}.{level}.txt
FILES_TO_DOWNLOAD = [
    "bacteria.WGS.blood.sample.clr.txt",
    "bacteria.WGS.blood.case.clr.txt",
    "bacteria.WGS.solid.sample.clr.txt",
    "bacteria.WGS.solid.case.clr.txt",
    "bacteria.WXS.blood.sample.clr.txt",
    "bacteria.WXS.blood.case.clr.txt",
    "bacteria.WXS.solid.sample.clr.txt",
    "bacteria.WXS.solid.case.clr.txt",
    "metadata.WGS.blood.sample.txt",
    "metadata.WGS.blood.case.txt",
    "metadata.WGS.solid.sample.txt",
    "metadata.WGS.solid.case.txt",
    "metadata.WXS.blood.sample.txt",
    "metadata.WXS.blood.case.txt",
    "metadata.WXS.solid.sample.txt",
    "metadata.WXS.solid.case.txt",
]

# Base URL for TCMA (This is a placeholder as direct links are dynamic/protected)
# Using a dummy URL to trigger the "Manual Download" message unless we find a stable one.
BASE_URL = "https://research.repository.duke.edu/downloads/files/" 

def download_file(url: str, dest_path: str):
    """Download a single file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded: {dest_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False

def setup_data_directory():
    """
    Create data directories and attempt to download files.
    """
    os.makedirs(RAW_INPUT_DIR, exist_ok=True)
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    missing_files = []
    
    for filename in FILES_TO_DOWNLOAD:
        file_path = os.path.join(RAW_INPUT_DIR, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
            
    if not missing_files:
        logger.info("All raw data files are present.")
        return

    logger.info(f"Missing {len(missing_files)} data files. Attempting download...")

    # NOTE: Since we don't have stable direct links for TCMA, 
    # we proceed to warn the user instructions.
    # If valid links were known, we would loop through missing_files and call download_file()
    
    logger.warning("!" * 80)
    logger.warning("COULD NOT AUTOMATICALLY DOWNLOAD DATASETS.")
    logger.warning("The TCMA dataset requires manual download due to repository restrictions.")
    logger.warning("Please download the following files:")
    for f in missing_files:
        logger.warning(f" - {f}")
    logger.warning(f"\nPlace them in: {RAW_INPUT_DIR}")
    logger.warning("!" * 80)
    
    # We don't raise an error here to allow the pipeline to proceed 
    # if the user only wants to run parts that don't need all files.
