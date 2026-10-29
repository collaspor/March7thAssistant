from pathlib import Path

import cv2
import numpy as np
import pytest

from tasks.power.reward_relic_flow import RewardProcessingError, process_rewards
from tasks.power.reward_relic_visual import recommendation_state


FIXTURES = Path(__file__).parents[1] / "fixtures" / "reward_relic"


def _image(name):
    return cv2.imdecode(np.fromfile(FIXTURES / name, dtype=np.uint8), cv2.IMREAD_COLOR)


class ScreenshotPage:
    def __init__(self):
        self.opened = []
        self.template = _image("thumb_template.png")

    def iter_cards(self):
        for index in range(1, 5):
            state = recommendation_state(_image(f"card_{index}.png"), self.template)
            yield index, "relic", state

    def open_card(self, card):
        self.opened.append(card)
        raise AssertionError("点赞和未知状态均不应打开详情")

    def traversal_complete(self):
        return True


def test_real_card_images_never_click_when_recommendation_is_unconfirmed():
    page = ScreenshotPage()
    with pytest.raises(RewardProcessingError, match="无法确认点赞标记"):
        process_rewards(page)
    assert page.opened == []
