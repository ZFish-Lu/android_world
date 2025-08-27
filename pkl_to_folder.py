import pickle
import json
import os
import gzip
from PIL import Image
import datetime
from tqdm import tqdm  # 新增


def pkl_to_folder(pkl_path, output_folder):
    """
    将pkl文件转换为包含图片和JSON的文件夹

    参数:
        pkl_path: pkl文件的路径
        output_folder: 输出文件夹路径
    """
    # 设置输出文件夹
    if output_folder is None:
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        output_folder = f"{base_name}_output"

    # 创建输出文件夹和图片子文件夹
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # 加载pkl文件
    try:
        with gzip.open(pkl_path, "rb") as f:  # 修改为支持pkl.gz
            data = pickle.load(f)
    except Exception as e:
        return

    # 处理数据 - 假设数据结构如你提供的示例
    # 提取主要数据（假设是列表中的第一个元素）
    if isinstance(data, list) and len(data) > 0:
        main_data = data[0]
    else:
        main_data = data

    # 分离图片和其他数据
    json_data = {}
    screenshots = []

    # 提取episode_data中的信息
    if "episode_data" in main_data:
        episode_data = main_data["episode_data"]

        # 仅当episode_data为dict时才处理截图
        if isinstance(episode_data, dict):
            if "screenshot" in episode_data and isinstance(
                episode_data["screenshot"], list
            ):
                screenshots = episode_data["screenshot"]
                # 移除截图数据，避免JSON序列化错误
                del episode_data["screenshot"]

            json_data["episode_data"] = episode_data
        else:
            # 跳过无法处理的episode_data
            json_data["episode_data"] = episode_data
    # 提取其他字段
    for key, value in main_data.items():
        if key != "episode_data":  # 已经处理过episode_data
            # 处理datetime对象，转换为字符串
            if isinstance(value, datetime.datetime):
                json_data[key] = value.isoformat()
            else:
                json_data[key] = value

    # 保存图片
    for i, img in enumerate(screenshots):
        if isinstance(img, Image.Image):
            # 获取对应的步骤编号
            step_num = (
                json_data["episode_data"]["step_number"][i]
                if "step_number" in json_data["episode_data"]
                else i
            )
            img_path = os.path.join(images_folder, f"step_{step_num}.png")
            img.save(img_path)

    # 保存JSON文件
    json_path = os.path.join(output_folder, "metadata.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存JSON文件失败: {e}")


def process_folder(input_folder, result_folder="./result"):
    """
    处理文件夹中的所有pkl.gz文件

    参数:
        input_folder: 包含pkl.gz文件的文件夹路径
        result_folder: 保存结果的根目录
    """
    # 获取输入文件夹的名称
    folder_name = os.path.basename(os.path.normpath(input_folder))
    result_folder = os.path.join(result_folder, folder_name)  # 添加子文件夹

    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)

    # 统计所有pkl.gz文件数量
    pkl_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pkl.gz"):
                pkl_files.append((root, file))

    for root, file in tqdm(pkl_files, desc="Processing pkl.gz files"):
        pkl_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, input_folder)
        output_folder = os.path.join(
            result_folder, relative_path, os.path.splitext(file)[0]
        )
        os.makedirs(output_folder, exist_ok=True)
        pkl_to_folder(pkl_path, output_folder)


if __name__ == "__main__":
    input_folder = "/home/luziyu/android_world/runs/run_20250827T164434817229"  # 替换为实际的输入文件夹路径
    process_folder(input_folder)
