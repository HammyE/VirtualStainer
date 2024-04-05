import os
import time


base_path = "runs"
print_files = True
changed_files = []
while True:
    files = os.listdir(base_path)
    for file in files:
        if print_files:
            print(file)
        # If file ends with .pt it is a model and needs to be named with the current time
        if file.endswith(".pt") and not file in changed_files:
            # wait for the file to be fully written
            print(f"Waiting for {file} to be fully written...")
            time.sleep(1)
            # Get the current time
            current_time = time.strftime("%Y%m%d-%H%M%S")
            # Split the file name into the name and the extension
            name, ext = os.path.splitext(file)
            # Rename the file with the current time
            os.rename(os.path.join(base_path, file), os.path.join(base_path, f"{name}_{current_time}{ext}"))

            changed_files.append(f"{name}_{current_time}{ext}")

            print(f"Renamed {file} to {name}_{current_time}{ext}")

    print_files = False