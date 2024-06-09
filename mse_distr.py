import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_all_datapoints(tensorboard_log_dir):
    # Initialize a dictionary to store the data
    data_dict = {}

    # Iterate through all event files in the log directory
    for root, _, files in os.walk(tensorboard_log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                file_path = os.path.join(root, file)
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                # Extract scalar summaries
                for tag in event_acc.Tags()["scalars"]:
                    events = event_acc.Scalars(tag)
                    if events:
                        if tag not in data_dict:
                            data_dict[tag] = []
                        for event in events:
                            data_dict[tag].append((event.step, event.value))

    return data_dict


data = extract_all_datapoints("runs_10/20240522-054727_Process-4/progress")

root = "runs_4"

master_data = {}

for run in os.listdir(root):
    if not os.path.isdir(f"{root}/{run}"):
        continue
    data = extract_all_datapoints(f"{root}/{run}/progress")
    master_data[run] = data

for measure in master_data[run].keys():
    print(master_data.keys())
    plt.figure(figsize=(10, 5))
    for run in master_data.keys():
        data = master_data[run]
        rolling_variance = pd.Series([value for _, value in data[measure]]).rolling(window=40).var()
        plt.plot(rolling_variance, label=run)
    plt.title(f"{measure} Variance")
    #plt.legend()
    plt.savefig(f"{root}/{measure}_variance.png")



