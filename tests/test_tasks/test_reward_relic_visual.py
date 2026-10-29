from pathlib import Path

import cv2
import numpy as np
import pytest

from tasks.power.reward_relic_visual import recommendation_state, discard_state


FIXTURES = Path(__file__).parents[1] / "fixtures" / "reward_relic"


def image(name):
    return cv2.imdecode(np.fromfile(FIXTURES / name, dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize("index, expected", [(1, "present"), (2, "present"), (3, "present"), (4, "unknown"), (5, "unknown"), (6, "unknown"), (7, "unknown")])
def test_recommendation_on_reward_cards(index, expected):
    assert recommendation_state(image(f"card_{index}.png"), image("thumb_template.png")) == expected


@pytest.mark.parametrize("name, expected", [("discard_off.png", "unmarked"), ("discard_on.png", "marked")])
def test_discard_icon_state(name, expected):
    assert discard_state(image(name)) == expected


def test_unrecognizable_discard_icon_is_unknown():
    assert discard_state(np.zeros((28, 26, 3), dtype=np.uint8)) == "unknown"


def test_plain_gray_area_is_not_an_unmarked_icon():
    assert discard_state(np.full((28, 26, 3), 160, dtype=np.uint8)) == "unknown"
