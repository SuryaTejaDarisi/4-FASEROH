import json
import random
import os

def split_taylor_dataset(input_path, train_path, test_path, train_size=11000, test_size=1000):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading data from {input_path}.....")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total objects found: {total_samples}")


    train_size = int(total_samples * 0.916) # Roughly 11/12
    test_size = total_samples - train_size

    shuffled_data = list(data)
    random.shuffle(shuffled_data) # Shuffle the Data
    train_set = shuffled_data[:train_size]
    test_set = shuffled_data[train_size:train_size + test_size]

    # Using separators=(',', ':') creates a compact single-line JSON 
    # if you want to keep the "single line" format strictly.
    print(f"Saving {len(train_set)} samples to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, separators=(',', ':'))

    print(f"Saving {len(test_set)} samples to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, separators=(',', ':'))

    print("Success! Dataset split complete.")

if __name__ == "__main__":
    SOURCE_FILE = "dataset.json" 
    TRAIN_FILE = "train.json"
    TEST_FILE = "test.json"
    split_taylor_dataset(SOURCE_FILE, TRAIN_FILE, TEST_FILE)