"""使用原项目自动化接口采集奖励图标，不发送鼠标或键盘输入。"""

import math

import cv2
import numpy as np

from .reward_relic_visual import discard_state, recommendation_state


class RewardRelicCapture:
    def __init__(self, automation):
        self.automation = automation

    def _capture_bgr(self, crop):
        if not isinstance(crop, (tuple, list)) or len(crop) != 4:
            raise ValueError("截图区域必须包含左、上、宽、高")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in crop):
            raise ValueError("截图区域必须为有限数字")
        x, y, width, height = crop
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("截图区域超出游戏窗口")
        image, _, _ = self.automation.take_screenshot(tuple(crop))
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def discard_state(self, crop):
        return discard_state(self._capture_bgr(crop))

    def recommendation_state(self, crop, template):
        rgb = np.asarray(template.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return recommendation_state(self._capture_bgr(crop), bgr)
