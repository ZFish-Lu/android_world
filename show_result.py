import os
import json

def calc_accuracy(root_dir):
    total = 0
    success_sum = 0
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        metadata_path = os.path.join(subdir_path, "metadata.json")
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            total += 1
            is_successful = data.get("is_successful", 0)
            try:
                is_successful = float(is_successful)
                if is_successful != is_successful:  # 检查nan
                    is_successful = 0
            except (ValueError, TypeError):
                is_successful = 0
            success_sum += is_successful
    if total == 0:
        print("No valid tasks found.")
        return
    accuracy = success_sum / total
    print(f"Total tasks: {total}")
    print(f"Sum of is_successful: {success_sum}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    path = "/data/project/android_world/result/v1_gpt4o+uitars"  # 替换为实际的结果文件夹路径
    calc_accuracy(path)
