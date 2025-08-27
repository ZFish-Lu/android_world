import ast
import base64
from io import BytesIO
import json
import math
import os
import logging
import re
from openai import OpenAI


# Configure logger to write to a file in logs directory with timestamped filename
import os
from datetime import datetime

logger = logging.getLogger("beap_agent.utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"beap_agent_utils_{timestamp}.log"), encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def call_llm(model, messages, max_tokens=1000, temperature=1.0, top_p=0.9):
    """
    模型调用，一般直接返回答案，若尝试3次均失败则返回None
    """
    try_times = 3
    if model.startswith("ui-tars"):
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://172.16.11.106:8027/v1",
        )
    elif model.startswith("qwen"):
        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    else:
        client = OpenAI(
            api_key=os.environ["EXTERNAL_API_KEY"],
            base_url="https://api5.xhub.chat/v1",
        )

    while True:
        if try_times <= 0:
            logger.info(
                f"Reach max retry times to fetch response from client, as error flag."
            )
            prediction = None
            break
        try:
            if model.startswith("gemini"):
                response = client.chat.completions.create(
                    model=model,
                    reasoning_effort="low",
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            prediction = response.choices[0].message.content

            # 过滤特殊字符，避免程序bug
            # prediction = re.sub(r'[^\x00-\x7F\u4e00-\u9fff.,!?，。！？：；、“”‘’]', '', prediction)
            # 在记录之前，从消息中滤除与图像相关的部分，提高logging的可读性
            filtered_messages = []
            for message in messages:
                if isinstance(message.get("content"), list):
                    filtered_content = []
                    for item in message["content"]:
                        if item.get("type") == "image_url":
                            filtered_content.append(
                                {"type": "image_url"}
                            )  # Retain type, remove image_url
                        else:
                            filtered_content.append(item)
                    filtered_messages.append({**message, "content": filtered_content})
                else:
                    filtered_messages.append(message)
            logger.info(f"LLM messages: {filtered_messages}")
            logger.info(f"LLM prediction: {prediction}\n")
            break
        except Exception as e:
            logger.info(
                f"Error when fetching response from client, with response: {response}"
            )
            prediction = None
            try_times -= 1

    return prediction


def image_preprocessing(image, max_pixels, min_pixels):
    obs_image_height = image.height
    obs_image_width = image.width
    if obs_image_height * obs_image_width > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))
    elif image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = math.ceil(image.width * resize_factor), math.ceil(
            image.height * resize_factor
        )
        image = image.resize((width, height))
    else:
        resize_factor = 1.0
        width, height = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode="eval")

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def extract_prediction_to_dict(s):
    pattern = r"<([a-zA-Z_][a-zA-Z0-9_-]*)>(.*?)</\1>"
    matches = re.findall(pattern, s, re.DOTALL)
    result = {tag.strip(): content.strip() for tag, content in matches}

    # 如果包含 exploration_methods，尝试将其解析为列表
    if "exploration_methods" in result:
        try:
            result["exploration_methods"] = json.loads(result["exploration_methods"])
        except json.JSONDecodeError:
            # 如果解析失败，保留原始字符串
            pass

    return result


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def convert_point_to_coordinates(text, is_answer=False):
    # 匹配 <bbox> 后面的四个数字
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        x = (x1 + x1) // 2  # 使用截断取整
        y = (y1 + y1) // 2  # 使用截断取整
        if is_answer:
            return f"({x},{y})"  # 只返回 (x, y) 格式
        return f"({x},{y})"  # 返回带标签的格式

    # 去掉 [EOS] 并替换 <bbox> 坐标
    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def parse_action_to_structure_output(
    text,
    factor,
    origin_resized_height,
    origin_resized_width,
    model_type="qwen25vl",
    max_pixels=16384 * 28 * 28,
    min_pixels=100 * 28 * 28,
):
    text = text.strip()

    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action: ")[-1]

    tmp_all_action = action_str.split(")\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            if not action_str.strip().endswith(")"):
                action_str = action_str.strip() + ")"

            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            if re.search(pattern, action_str):  # 检查是否有匹配项
                content = re.sub(pattern, escape_quotes, action_str)
            else:
                raise ValueError("Pattern not found in the input string.")

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        if not action_str.strip().endswith(")"):
            action_str = action_str.strip() + ")"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num / smart_resize_height))
                        else:
                            float_numbers.append(float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0],
                        float_numbers[1],
                        float_numbers[0],
                        float_numbers[1],
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append(
            {
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text,
            }
        )
    return actions


def parse_structure_output_to_json_action(
    structure_output, image_height: int, image_width: int
):
    json_action = {"action_type": structure_output["action_type"]}

    if structure_output["action_type"] in ["click", "long_press", "scroll", "drag"]:
        start_box = structure_output["action_inputs"].get("start_box", None)
        start_box = str(start_box)
        if start_box:
            start_box = eval(start_box)
            if len(start_box) == 4:
                x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
            elif len(start_box) == 2:
                x1, y1 = start_box
                x2 = x1
                y2 = y1
            x = round(float((x1 + x2) / 2) * image_width, 3)
            y = round(float((y1 + y2) / 2) * image_height, 3)
            json_action["x"] = x
            json_action["y"] = y

        if structure_output["action_type"] == "scroll":
            direction = structure_output["action_inputs"].get("direction", "down")
            json_action["direction"] = direction
        
        if structure_output["action_type"] == "drag":
            json_action["action_type"] = "scroll"
            json_action["direction"] = "down"

    elif structure_output["action_type"] == "type":
        json_action["action_type"]=  "input_text"
        text = structure_output["action_inputs"].get("content", "")
        json_action["text"] = text

    elif structure_output["action_type"] == "open_app":
        app_name = structure_output["action_inputs"].get("app_name", "")
        json_action["app_name"] = app_name

    elif structure_output["action_type"] == "press_home":
        json_action["action_type"] = "navigate_home"

    elif structure_output["action_type"] == "press_back":
        json_action["action_type"] = "navigate_back"

    return json_action


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action
            )

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'",
                )
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


def extract_value_from_output(output: str, tag: str) -> str:
    """
    Extracts the value enclosed within a specific tag from the given output string.

    Args:
        output (str): The string containing the output.
        tag (str): The tag to extract the value for.

    Returns:
        str: The extracted value, or an empty string if the tag is not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, output, re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_semantic_action_to_json_action(text):
    """
    解析字符串中的动作指令，支持以下格式：
    1. open_app(app_name="xxx") -> {"action_type": "open_app", "app_name": "xxx"}
    2. press_home() -> {"action_type": "navigate_home"}
    3. press_back() -> {"action_type": "navigate_back"}
    
    Args:
        text (str): 要解析的字符串
    
    Returns:
        dict or None: 解析结果字典，如果没有匹配则返回None
    """
    if not isinstance(text, str):
        return None
    
    # 去除字符串前后的空白字符
    text = text.strip()
    
    # 1. 匹配 open_app(app_name="xxx") 格式
    open_app_pattern = r'open_app\s*\(\s*app_name\s*=\s*["\']([^"\']*)["\']\s*\)'
    match = re.match(open_app_pattern, text)
    if match:
        app_name = match.group(1)
        return {
            "action_type": "open_app",
            "app_name": app_name
        }
    
    # 2. 匹配 press_home() 格式
    home_pattern = r'press_home\s*\(\s*\)'
    if re.match(home_pattern, text):
        return {
            "action_type": "navigate_home"
        }
    
    # 3. 匹配 press_back() 格式
    back_pattern = r'press_back\s*\(\s*\)'
    if re.match(back_pattern, text):
        return {
            "action_type": "navigate_back"
        }
    
    # 如果都不匹配，返回None
    return None