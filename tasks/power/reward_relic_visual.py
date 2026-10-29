"""结算遗器图标的实验性识别；未接入游戏操作流程。"""

import cv2
import numpy as np


def recommendation_state(card, template):
    """在受限的已验证样本尺寸下识别点赞，其余画面返回未知。"""
    if (not isinstance(card, np.ndarray) or not isinstance(template, np.ndarray)
            or card.ndim != 3 or template.ndim != 3 or card.shape[2] != 3
            or template.shape[2] != 3 or card.dtype != np.uint8 or template.dtype != np.uint8):
        return "unknown"
    if template.size == 0 or float(np.std(template)) < 1:
        return "unknown"
    height, width = card.shape[:2]
    if height < 30 or width < 45:
        return "unknown"
    region = card[:round(height * 0.5), round(width * 0.60):]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return "unknown"
    match = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    score = float(np.max(match))
    if not np.isfinite(score):
        return "unknown"
    if score >= 0.8:
        return "present"
    if card.shape != (61, 74, 3) or template.shape != (24, 23, 3):
        return "unknown"
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    golden = cv2.countNonZero(cv2.inRange(hsv, (0, 40, 70), (40, 255, 255)))
    edges = np.count_nonzero(cv2.Canny(card, 60, 120))
    if score <= 0.50 and golden >= 2000 and edges >= 500:
        return "absent"
    return "unknown"


def discard_state(icon):
    """识别样本中红色弃置图标；其他状态保持未知。"""
    if (not isinstance(icon, np.ndarray) or icon.ndim != 3 or icon.shape[2] != 3
            or icon.dtype != np.uint8 or min(icon.shape[:2]) < 20):
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
