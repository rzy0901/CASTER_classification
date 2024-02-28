import os
import shutil

def copy_files_with_suffix(src_folder, dest_folder, suffix="_r.jpg"):
    if not os.path.exists(src_folder):
        print(f"Error: Source folder '{src_folder}' not found.")
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(suffix):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_folder)
                dest_file_path = os.path.join(dest_folder, relative_path)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} to {dest_file_path}")

if __name__ == "__main__":
    copy_files_with_suffix("./model2/train/", "./train_target_3x10/")
    copy_files_with_suffix("./model3/train/", "./train_target_3x20/")
    copy_files_with_suffix("./model4/train/", "./train_target_3x30/")
