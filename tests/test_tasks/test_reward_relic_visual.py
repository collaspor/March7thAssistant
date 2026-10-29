from pathlib import Path

import cv2
import numpy as np
import pytest

from tasks.power.reward_relic_visual import recommendation_state, discard_state


FIXTURES = Path(__file__).parents[1] / "fixtures" / "reward_relic"


def image(name):
    return cv2.imdecode(np.fromfile(FIXTURES / name, dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize("index, expected", [(1, "present"), (2, "present"), (3, "present"), (4, "absent"), (5, "absent"), (6, "absent"), (7, "absent"), (8, "absent"), (9, "absent")])
def test_recommendation_on_reward_cards(index, expected):
    assert recommendation_state(image(f"card_{index}.png"), image("thumb_template.png")) == expected


@pytest.mark.parametrize("name, expected", [("discard_off.png", "unmarked"), ("discard_on.png", "marked")])
def test_discard_icon_state(name, expected):
    assert discard_state(image(name)) == expected


def test_unrecognizable_discard_icon_is_unknown():
    assert discard_state(np.zeros((28, 26, 3), dtype=np.uint8)) == "unknown"


def test_plain_gray_area_is_not_an_unmarked_icon():
    assert discard_state(np.full((28, 26, 3), 160, dtype=np.uint8)) == "unknown"


def test_blank_template_does_not_count_as_a_recommendation():
    assert recommendation_state(image("card_4.png"), np.zeros((24, 23, 3), dtype=np.uint8)) == "unknown"


def test_unrecognizable_card_cannot_be_classified_as_absent():
    assert recommendation_state(np.zeros((61, 74, 3), dtype=np.uint8), image("thumb_template.png")) == "unknown"


@pytest.mark.parametrize("start_x", [54, 57])
def test_partially_obscured_recommendation_is_not_absent(start_x):
    card = image("card_1.png")
    card[2:25, start_x:73] = card[30:53, start_x:73]
    assert recommendation_state(card, image("thumb_template.png")) == "unknown"


def test_plain_red_area_is_not_a_discard_icon():
    image_red = np.full((28, 26, 3), (0, 0, 220), dtype=np.uint8)
    assert discard_state(image_red) == "unknown"


def test_malformed_images_are_unknown():
    malformed = np.random.default_rng(1).integers(0, 256, (28, 26, 2), dtype=np.uint8)
    assert discard_state(malformed) == "unknown"
    assert recommendation_state(image("card_1.png"), malformed) == "unknown"
