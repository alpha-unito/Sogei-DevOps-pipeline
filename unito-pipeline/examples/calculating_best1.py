import os
import shutil
import pandas as pd
import glob

def save_best_model():
    out_folder = os.getcwd()  # use current working directory
    print(f"Running in: {out_folder}")

    # 1. Create "Final_models" folder
    final_models_dir = os.path.join(out_folder, "Final_models")
    os.makedirs(final_models_dir, exist_ok=True)

    # 2. Find all CSV files starting with "metrics"
    csv_files = glob.glob(os.path.join(out_folder, "metrics*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files starting with 'metrics' found in {out_folder}")

    best_iou = None
    best_run_name = None
    best_csv = None

    # 3. Iterate over CSVs and track the run with the lowest IoU
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "iou" not in df.columns:
            print(f"Skipping {csv_file}, no 'iou' column.")
            continue

        min_idx = df["iou"].idxmin()
        min_iou = df.loc[min_idx, "iou"]

        run_name = df.loc[min_idx, "run"] if "run" in df.columns else os.path.splitext(os.path.basename(csv_file))[0]

        if best_iou is None or min_iou < best_iou:
            best_iou = min_iou
            best_run_name = run_name
            best_csv = csv_file

    if best_run_name is None:
        raise ValueError("No valid IoU values found in CSV files!")

    print(f"Best run (lowest IoU {best_iou}): {best_run_name} from {best_csv}")

    # 4. Find the folder in out_folder having the run name in it
    matching_folders = [
        f for f in os.listdir(out_folder)
        if os.path.isdir(os.path.join(out_folder, f)) and best_run_name in f
    ]
    if not matching_folders:
        raise FileNotFoundError(f"No folder found containing '{best_run_name}' in {out_folder}")

    best_folder = os.path.join(out_folder, matching_folders[0])
    print(f"Found folder: {best_folder}")

    # 5. Copy best.pt from folder/state/ to Final_models
    state_file = os.path.join(best_folder, "state", "best.pt")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"{state_file} not found!")

    dest_file = os.path.join(final_models_dir, f"{matching_folders[0]}_best.pt")
    shutil.copy(state_file, dest_file)
    print(f"Copied {state_file} to {dest_file}")


if __name__ == "__main__":
    save_best_model()