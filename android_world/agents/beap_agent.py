import logging
import math
import time
from android_world.agents import base_agent
from android_world.agents.beap_utils import (
    add_box_token,
    extract_value_from_output,
    image_preprocessing,
    call_llm,
    parse_action_to_structure_output,
    pil_to_base64,
    parse_structure_output_to_json_action,
)
from android_world.agents.beap_prompt import (
    PLANNER_PROMPT,
    REPLAN_PLANNER_PROMPT,
    TRACKER_PROMPT,
    TRACKER_PROMPT_NO_BACKTRACK_STATUS,
    EXECUTOR_MOBILE_PROMPT,
    BACKTRACK_EXECUTOR_MOBILE_PROMPT,
    BACKTRACK_TRACKER_PROMPT,
    SUMMARY_PROMPT,
    UITARS_MOBILE_ACTION_SPACE,
)
from android_world.env import interface
from android_world.env import json_action
from PIL import Image


# Configure logger to write to a file in logs directory with timestamped filename
import os
from datetime import datetime

logger = logging.getLogger("beap_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"beap_agent_{timestamp}.log"), encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


class BEAPAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "BEAPAgent",
        wait_after_action_seconds: float = 2.0,
        planner_model: str = "gpt-4o",
        executor_model: str = "ui-tars",
        history_n: int = 3,
    ):
        super().__init__(env, name)
        self.wait_after_action_seconds = wait_after_action_seconds
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.history_n = history_n

        self.action_step = 0  # 统计步数
        self.is_backtrack_status = False
        self.need_replan = False
        self.plan = ""
        self.max_backtrack_times = 5
        self.max_backtrack_steps = 5
        self.invalid_ways = []
        self.history_images = []  # 存储的是PIL.Image格式，仅在正常状态下记录
        self.history_grounding_actions = []  # executor的输出

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()

        self.action_step = 0  # 统计步数
        self.is_backtrack_status = False
        self.need_replan = False
        self.plan = ""
        self.max_backtrack_times = 5
        self.max_backtrack_steps = 5
        self.invalid_ways = []
        self.history_images = []  # 存储的是PIL.Image格式，仅在正常状态下记录
        self.history_grounding_actions = []  # executor的输出

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {"semantic_action": None, "action": None, "screenshot": None, "is_backtrack_status": None}
        self.action_step += 1
        logger.info("----------step %s----------", str(self.action_step))

        if self.action_step >= 50:
            logger.info("Action step exceeded maximum limit, marking as failed.")
            step_data["semantic_action"] = "FAIL"
            step_data["action"] = {
                "action_type": "status",
                "goal_status": "infeasible",
            }
            step_data["is_backtrack_status"] = False
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        # 统一获取桌面截图并转为PIL图像
        state = self.get_post_transition_state()
        image_numpy = state.pixels.copy()
        image = Image.fromarray(image_numpy)
        obs_image_height = image.height
        obs_image_width = image.width
        image = image_preprocessing(image, MAX_PIXELS, MIN_PIXELS)
        self.history_images.append(image)
        step_data["screenshot"] = image

        if self.plan == "":
            # 第一次执行，history初始化
            self.history_images.append(image)
            self.history_grounding_actions.append(
                "No operation is performed, only the starting status is recorded"
            )
            self.plan = self.planner(task=goal, is_replan=False)

        # agent核心逻辑处理
        while True:
            if self.need_replan:
                # 需要重新规划，并清理history
                self.plan = self.planner(task=goal, is_replan=True)
                self.history_images = self.history_images[:-2]
                self.history_grounding_actions = self.history_grounding_actions[:-1]
                self.history_images.append(image)
                self.need_replan = False

            if self.is_backtrack_status:
                is_backtracked, semantic_action = self.backtrack_tracker()
                if is_backtracked == "True":
                    logger.info("Backtracking completed, resetting state.")
                    self.is_backtrack_status = False
                    self.need_replan = True
                    self.max_backtrack_steps = 5
                elif self.max_backtrack_steps <= 0:
                    logger.info(
                        "Backtracking failed, max backtrack steps reached, resetting state."
                    )
                    self.is_backtrack_status = False
                    self.need_replan = True
                    self.max_backtrack_steps = 5
                else:
                    logger.info(f"Backtracking in progress")
                    prediction, grounding_actions = self.executor(
                        task=goal,
                        semantic_action=semantic_action,
                        obs_image_height=obs_image_height,
                        obs_image_width=obs_image_width,
                    )
                    self.max_backtrack_steps -= 1
                    self.history_images = self.history_images[:-1]
                    step_data["semantic_action"] = semantic_action
                    step_data["action"] = grounding_actions
                    try:
                        for grounding_action in grounding_actions:
                            self.env.execute_action(
                                json_action.JSONAction(**grounding_action)
                            )
                            time.sleep(self.wait_after_action_seconds)
                    except Exception as e:
                        logging.warning(
                            f"Failed to execute action: {grounding_action}. Error: {e}"
                        )
                        step_data["action"] = "Error"
                    step_data["is_backtrack_status"] = True
                    return base_agent.AgentInteractionResult(
                        False,
                        step_data,
                    )

            else:
                plan, exploration_status, semantic_action, answer = self.tracker(
                    task=goal
                )
                self.plan = plan

                if exploration_status == "CONTINUE":
                    prediction, grounding_actions = self.executor(
                        task=goal,
                        semantic_action=semantic_action,
                        obs_image_height=obs_image_height,
                        obs_image_width=obs_image_width,
                    )
                    self.history_grounding_actions.append(prediction)
                    step_data["semantic_action"] = semantic_action
                    step_data["action"] = grounding_actions
                    try:
                        for grounding_action in grounding_actions:
                            self.env.execute_action(
                                json_action.JSONAction(**grounding_action)
                            )
                            time.sleep(self.wait_after_action_seconds)
                    except Exception as e:
                        logging.warning(
                            f"Failed to execute action: {grounding_action}. Error: {e}"
                        )
                        step_data["action"] = "Error"
                    step_data["is_backtrack_status"] = False
                    return base_agent.AgentInteractionResult(
                        False,
                        step_data,
                    )

                elif exploration_status == "BACKTRACK":
                    logger.info(f"Entering backtrack status.")
                    invalid_way = self.summarize_invalid_way(task=goal)
                    self.invalid_ways.append(invalid_way)
                    self.is_backtrack_status = True
                    self.max_backtrack_times -= 1

                elif exploration_status == "FAIL":
                    step_data["semantic_action"] = "FAIL"
                    step_data["action"] = {
                        "action_type": "status",
                        "goal_status": "infeasible",
                    }
                    step_data["is_backtrack_status"] = False
                    return base_agent.AgentInteractionResult(
                        True,
                        step_data,
                    )

                elif exploration_status == "DONE":
                    step_data["semantic_action"] = "DONE"
                    step_data["action"] = {
                        "action_type": "status",
                        "goal_status": "complete",
                    }
                    step_data["is_backtrack_status"] = False
                    return base_agent.AgentInteractionResult(
                        True,
                        step_data,
                    )

                elif exploration_status == "ANSWER":
                    grounding_action = {
                        "action_type": "answer",
                        "text": answer,
                    }
                    step_data["semantic_action"] = grounding_action
                    step_data["action"] = grounding_action
                    self.env.execute_action(json_action.JSONAction(**grounding_action))
                    time.sleep(self.wait_after_action_seconds)
                    step_data["is_backtrack_status"] = False
                    return base_agent.AgentInteractionResult(
                        True,
                        step_data,
                    )

    def planner(self, task, is_replan=False):
        if is_replan:
            logger.info(f"Replanning with invalid_ways: {self.invalid_ways}")
            prompt = REPLAN_PLANNER_PROMPT.format(
                task=task, plan=self.plan, invalid_ways=self.invalid_ways
            )
        else:
            logger.info(f"planner is called.\n")
            prompt = PLANNER_PROMPT.format(task=task)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        encoded_string = pil_to_base64(self.history_images[-1])
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
            }
        )
        while True:
            prediction = call_llm(self.planner_model, messages)
            plan = extract_value_from_output(prediction, "plan")
            if plan != "":
                return plan

    def executor(self, task, semantic_action, obs_image_height, obs_image_width):
        logger.info(f"Executor is called.\n")
        origin_resized_height = self.history_images[-1].height
        origin_resized_width = self.history_images[-1].width

        if self.is_backtrack_status:
            prompt = BACKTRACK_EXECUTOR_MOBILE_PROMPT.format(
                action_space=UITARS_MOBILE_ACTION_SPACE,
                semantic_action=semantic_action,
            )
        else:
            prompt = EXECUTOR_MOBILE_PROMPT.format(
                action_space=UITARS_MOBILE_ACTION_SPACE,
                task=task,
                semantic_action=semantic_action,
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        if self.is_backtrack_status == False:
            start_idx = max(0, len(self.history_grounding_actions) - self.history_n)
            for history_idx in range(start_idx, len(self.history_grounding_actions)):
                encoded_string = pil_to_base64(self.history_images[history_idx])
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_string}"
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": add_box_token(
                                    self.history_grounding_actions[history_idx]
                                ),
                            }
                        ],
                    }
                )

        encoded_string = pil_to_base64(self.history_images[-1])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }
                ],
            }
        )

        prediction = call_llm(self.executor_model, messages)
        parsed_responses = parse_action_to_structure_output(
            prediction,
            1000,
            origin_resized_height,
            origin_resized_width,
        )

        grounding_actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:
                if parsed_response["action_type"] == FINISH_WORD:
                    return prediction, [{"action_type": "wait"}]

            # 解析动作为android_world可执行的格式
            grounding_actions.append(
                parse_structure_output_to_json_action(
                    parsed_response, obs_image_height, obs_image_width
                )
            )

        return prediction, grounding_actions

    def tracker(self, task):
        logger.info(f"Tracker is called.\n")
        if self.max_backtrack_times <= 0 or len(self.history_grounding_actions) <= 1:
            logger.info(
                f"Max backtrack times reached or no history available, use TRACKER_PROMPT_NO_BACKTRACK_STATUS."
            )
            prompt = TRACKER_PROMPT_NO_BACKTRACK_STATUS.format(
                task=task, plan=self.plan, invalid_ways=self.invalid_ways
            )
        else:
            prompt = TRACKER_PROMPT.format(
                task=task, plan=self.plan, invalid_ways=self.invalid_ways
            )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        start_idx = max(0, len(self.history_grounding_actions) - self.history_n)
        for history_idx in range(start_idx, len(self.history_grounding_actions)):
            encoded_string = pil_to_base64(self.history_images[history_idx])
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_string}"
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"History grounding action: {self.history_grounding_actions[history_idx]}",
                        }
                    ],
                }
            )
        encoded_string = pil_to_base64(self.history_images[-1])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }
                ],
            }
        )

        # 校验输出是否正确无误
        while True:
            prediction = call_llm(self.planner_model, messages)
            plan = extract_value_from_output(prediction, "plan")
            exploration_status = extract_value_from_output(
                prediction, "exploration_status"
            )
            semantic_action = extract_value_from_output(prediction, "semantic_action")
            answer = extract_value_from_output(prediction, "answer")

            if plan != "" and exploration_status != "":
                if (
                    exploration_status == "CONTINUE"
                    and semantic_action != ""
                    or exploration_status == "BACKTRACK"
                    or exploration_status == "DONE"
                    or exploration_status == "FAIL"
                    or exploration_status == "ANSWER"
                    and answer != ""
                ):
                    break
            logger.warning(f"The output does not meet the requirements. Re-output")

        return plan, exploration_status, semantic_action, answer

    def backtrack_tracker(self):
        logger.info(f"Backtrack Tracker is called.\n")
        prompt = BACKTRACK_TRACKER_PROMPT.format(
            history_actions=self.history_grounding_actions[-1],
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        encoded_string_target = pil_to_base64(self.history_images[-2])
        encoded_string_current = pil_to_base64(self.history_images[-1])
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "target screenshot:"}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string_target}"
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "current screenshot:"}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string_current}"
                        },
                    }
                ],
            }
        )

        # 校验输出是否正确无误
        try_times = 3
        while try_times > 0:
            prediction = call_llm(self.planner_model, messages)
            is_backtracked = extract_value_from_output(prediction, "is_backtracked")
            semantic_action = extract_value_from_output(prediction, "semantic_action")
            if is_backtracked == "True" or (
                is_backtracked == "False" and semantic_action != ""
            ):
                break
            logger.warning(f"The output does not meet the requirements. Re-output")
            try_times -= 1
            if try_times <= 0:
                raise ValueError("Failed to get a valid response after 3 attempts.")

        return is_backtracked, semantic_action

    def summarize_invalid_way(self, task):
        prompt = SUMMARY_PROMPT.format(task=task, plan=self.plan)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        encoded_string = pil_to_base64(self.history_images[-2])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"History action: {self.history_grounding_actions[-1]}",
                    }
                ],
            }
        )
        encoded_string = pil_to_base64(self.history_images[-1])
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                    }
                ],
            }
        )
        prediction = call_llm(self.planner_model, messages)
        return prediction
