import json
import random
import os

def split_taylor_dataset(input_path, train_path, test_path, train_size=11000, test_size=1000):
    """
    Reads a JSON file containing a list of dataset objects and splits them 
    randomly into training and testing files.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load the full dataset
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total objects found: {total_samples}")

    if total_samples < (train_size + test_size):
        print(f"Warning: Dataset only has {total_samples} objects. Adjusting split...")
        # Optional: handle cases where you have fewer than 12k
        train_size = int(total_samples * 0.916) # Roughly 11/12
        test_size = total_samples - train_size

    # 2. Shuffle the data randomly
    # We use a copy to avoid mutating the original 'data' list if used elsewhere
    shuffled_data = list(data)
    random.shuffle(shuffled_data)

    # 3. Split the list
    train_set = shuffled_data[:train_size]
    test_set = shuffled_data[train_size:train_size + test_size]

    # 4. Save to files
    # Note: Using separators=(',', ':') creates a compact single-line JSON 
    # if you want to keep the "single line" format strictly.
    print(f"Saving {len(train_set)} samples to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, separators=(',', ':'))

    print(f"Saving {len(test_set)} samples to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, separators=(',', ':'))

    print("Success! Dataset split complete.")

if __name__ == "__main__":
    # Update these paths to match your local file names
    SOURCE_FILE = "dataset.json" 
    TRAIN_FILE = "train.json"
    TEST_FILE = "test.json"

    split_taylor_dataset(SOURCE_FILE, TRAIN_FILE, TEST_FILE)