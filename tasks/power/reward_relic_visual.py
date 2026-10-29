"""结算遗器图标的实验性识别；未接入游戏操作流程。"""

import cv2
import numpy as np


def recommendation_state(card, template):
    """只确认点赞标记；未匹配或画面不完整时返回未知。"""
    if card is None or template is None or card.ndim != 3 or template.ndim != 3:
        return "unknown"
    height, width = card.shape[:2]
    if height < 30 or width < 45:
        return "unknown"
    region = card[:round(height * 0.5), round(width * 0.60):]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return "unknown"
    match = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(match))
    return "present" if np.isfinite(score) and score >= 0.8 else "unknown"


def discard_state(icon):
    """识别样本中红色弃置图标；其他状态保持未知。"""
    if icon is None or icon.ndim != 3 or min(icon.shape[:2]) < 20:
        return "unknown"
    if np.count_nonzero(cv2.Canny(icon, 50, 100)) < 40:
        return "unknown"
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))
    if cv2.countNonZero(red) >= 40:
        return "marked"
    gray = cv2.inRange(hsv, (0, 0, 80), (179, 60, 240))
    if cv2.countNonZero(gray) >= 500:
        return "unmarked"
    return "unknown"
