from pathlib import Path

import pytest
from PIL import Image

from tasks.power.reward_relic_capture import RewardRelicCapture


FIXTURES = Path(__file__).parents[1] / "fixtures" / "reward_relic"


class FakeAutomation:
    def __init__(self, image):
        self.image = image
        self.crops = []

    def take_screenshot(self, crop):
        self.crops.append(crop)
        return self.image, (10, 20, *self.image.size), 1


def test_discard_state_uses_original_screenshot_interface():
    auto = FakeAutomation(Image.open(FIXTURES / "discard_on.png").convert("RGB"))
    assert RewardRelicCapture(auto).discard_state((0.1, 0.2, 0.03, 0.04)) == "marked"
    assert auto.crops == [(0.1, 0.2, 0.03, 0.04)]


def test_invalid_crop_does_not_capture_or_guess():
    auto = FakeAutomation(Image.open(FIXTURES / "discard_off.png").convert("RGB"))
    with pytest.raises(ValueError):
        RewardRelicCapture(auto).discard_state((0.9, 0.2, 0.2, 0.1))
    assert not auto.crops


def test_recommendation_uses_captured_rgb_image():
    auto = FakeAutomation(Image.open(FIXTURES / "card_2.png").convert("RGB"))
    template = Image.open(FIXTURES / "thumb_template.png").convert("RGB")
    assert RewardRelicCapture(auto).recommendation_state((0.1, 0.1, 0.2, 0.2), template) == "present"
