import pickle
import pydicom as dicomio
import os
from collections import Counter
import zipfile

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def loadFileInformation(filename):
    ''' Extract and return metadata from a DICOM file, such as the SOPInstanceUID.'''
    
    information = {}
    ds = dicomio.read_file(filename, force=True)
    information['dicom_num'] = ds.SOPInstanceUID
    # information['PatientID'] = ds.PatientID
    # information['PatientName'] = ds.PatientName
    # information['PatientBirthDate'] = ds.PatientBirthDate
    # information['PatientSex'] = ds.PatientSex
    # information['StudyID'] = ds.StudyID
    # information['StudyDate'] = ds.StudyDate
    # information['StudyTime'] = ds.StudyTime
    # information['InstitutionName'] = ds.InstitutionName
    # information['Manufacturer'] = ds.Manufacturer
    # information['NumberOfFrames'] = ds.NumberOfFrames
    return information

def rename_files_in_folder(folder_path, patient_to_histology, histology_prefix):
    """Renames files in the folder by adding the correct histology letter prefix."""
    for filename in os.listdir(folder_path):
        # Extract patient ID from the file name (first part before "_")
        patient_id = filename.split("_")[0]

        # Get corresponding histology and prefix
        histology = patient_to_histology.get(patient_id, None)
        if histology in histology_prefix:
            prefix = histology_prefix[histology]
            new_filename = f"{prefix}_{filename}"

            # Rename file
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
            

def count_files_by_prefix(folder_path, prefixes):
    """Counts how many files in a folder start with each prefix."""
    counts = Counter()
    
    for filename in os.listdir(folder_path):
        for prefix in prefixes:
            if filename.startswith(prefix):
                counts[prefix] += 1

    return counts


def update_label_files(label_dir, histology_class_map):
    """Updates YOLO label files by replacing the first number with the correct class."""
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            histology_prefix = filename[0]  # Extract the first letter
            
            if histology_prefix in histology_class_map:
                class_number = histology_class_map[histology_prefix]
                
                label_path = os.path.join(label_dir, filename)
                
                # Read and modify the label file
                with open(label_path, "r") as f:
                    lines = f.readlines()

                # Replace first number in each line
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Ensure correct YOLO format
                        parts[0] = class_number  # Replace class ID
                        updated_lines.append(" ".join(parts))

                # Write updated lines back to the file
                with open(label_path, "w") as f:
                    f.write("\n".join(updated_lines) + "\n")

                print(f"Updated: {filename}")
                
def zip_directory(directory_path, output_zip_path=None):
    """
    Zips the contents of a directory into a zip file.

    :param directory_path: Path to the directory to be zipped.
    :param output_zip_path: Path to save the output zip file. If None, saves in the same directory.
    :return: Path to the created zip file.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Error: The directory '{directory_path}' does not exist or is not a directory.")

    # Default zip file name based on the directory name
    if output_zip_path is None:
        output_zip_path = os.path.join(os.path.dirname(directory_path), f"{os.path.basename(directory_path)}.zip")

    # Create the zip file
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Preserve folder structure inside the zip
                arcname = os.path.relpath(file_path, start=directory_path)
                zipf.write(file_path, arcname)

    print(f"Directory '{directory_path}' successfully zipped to '{output_zip_path}'")
    return output_zip_path