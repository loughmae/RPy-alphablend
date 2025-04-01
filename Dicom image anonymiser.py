import os
import pydicom
import tkinter as tk
from tkinter import filedialog

TAGS_TO_ANONYMIZE = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0032),  # PatientBirthTime
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1005),  # PatientBirthName
    (0x0010, 0x1060),  # PatientMotherBirthName
    (0x0008, 0x0050),  # AccessionNumber
    (0x0020, 0x0010),  # StudyID
    (0x0008, 0x0020),  # StudyDate
    (0x0008, 0x0030),  # StudyTime
    (0x0008, 0x0021),  # SeriesDate
    (0x0008, 0x0031),  # SeriesTime
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x0092),  # ReferringPhysicianAddress
    (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumbers
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1052),  # PerformingPhysicianName
]


def anonymize_and_save(ds, output_path):
    try:
        # Use SeriesName if present
        series_name = ds.get((0x0008, 0x103E)) or "SeriesName"
        for tag in TAGS_TO_ANONYMIZE:
            if tag in ds:
                # Use the series name for PatientName, generic for others
                if tag == (0x0010, 0x0010):
                    ds[tag].value = "Anonymized"
                else:
                    ds[tag].value = "Anonymized"

        ds.remove_private_tags()

        # Save using pydicom
        pydicom.dcmwrite(output_path, ds)
        return True

    except Exception as e:
        print(f"[Error in anonymize_and_save] {output_path} -> {e}")
        return False


def main():
    root = tk.Tk()
    root.withdraw()

    input_folder = filedialog.askdirectory(title="Select the folder containing files")
    if not input_folder:
        print("No input folder selected.")
        return

    output_folder = filedialog.askdirectory(title="Select the output folder for anonymized files")
    if not output_folder:
        print("No output folder selected.")
        return

    total_files = 0
    dicom_readable_count = 0
    anonymized_count = 0

    for root_dir, _, files in os.walk(input_folder):
        for filename in files:
            file_path = os.path.join(root_dir, filename)
            total_files += 1

            try:
                ds = pydicom.dcmread(file_path)
                dicom_readable_count += 1
            except Exception as e:
                print(f"[Not a valid DICOM] {file_path} -> {e}")
                continue  # Not a DICOM or reading error

            # Create mirrored folder structure
            rel_path = os.path.relpath(root_dir, input_folder)
            out_dir = os.path.join(output_folder, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, filename)
            success = anonymize_and_save(ds, out_path)
            if success:
                anonymized_count += 1
            else:
                print(f"[Failed to anonymize] {file_path}")

    print(f"Total files in folder: {total_files}")
    print(f"Files identified as valid DICOM: {dicom_readable_count}")
    print(f"Total anonymized: {anonymized_count}")


if __name__ == "__main__":
    main()
