import pytest

from tasks.power.reward_relic_flow import RewardProcessingError, process_rewards


class FakeRewardPage:
    def __init__(self, cards, details=None, after_click="marked"):
        self.cards = cards
        self.details = details or {}
        self.after_click = after_click
        self.opened = []
        self.marked = []
        self.closed = []

    def iter_cards(self):
        yield from self.cards

    def open_card(self, card):
        self.opened.append(card[0])
        return self.details.get(card[0], "unmarked")

    def mark_discard(self, card):
        self.marked.append(card[0])
        return self.after_click

    def close_card(self, card):
        self.closed.append(card[0])
        return True

    def traversal_complete(self):
        return True


def test_marks_only_confirmed_unrecommended_relics_and_keeps_recommended():
    page = FakeRewardPage([("a", "relic", "present"), ("b", "relic", "absent"), ("c", "other", "unknown")])
    assert process_rewards(page) == 1
    assert page.opened == ["b"]
    assert page.marked == ["b"]
    assert page.closed == ["b"]


def test_already_marked_is_not_clicked_again():
    page = FakeRewardPage([("a", "relic", "absent")], {"a": "marked"})
    assert process_rewards(page) == 0
    assert page.marked == []
    assert page.closed == ["a"]


def test_unknown_recommendation_stops_without_clicking():
    page = FakeRewardPage([("a", "relic", "unknown"), ("b", "relic", "absent")])
    with pytest.raises(RewardProcessingError):
        process_rewards(page)
    assert page.opened == []
    assert page.marked == []


@pytest.mark.parametrize("detail, result", [("unknown", "marked"), ("unmarked", "unknown")])
def test_uncertain_detail_or_failed_verification_stops_processing(detail, result):
    page = FakeRewardPage([("a", "relic", "absent"), ("b", "relic", "absent")], {"a": detail}, result)
    with pytest.raises(RewardProcessingError):
        process_rewards(page)
    assert page.opened == ["a"]
    assert page.closed == []


def test_overlapping_scroll_pages_do_not_toggle_a_card_twice():
    page = FakeRewardPage([("a", "relic", "absent"), ("a", "relic", "absent"), ("b", "relic", "absent")])
    assert process_rewards(page) == 2
    assert page.marked == ["a", "b"]


def test_unknown_item_type_stops_without_opening_it():
    page = FakeRewardPage([("a", "unknown", "absent")])
    with pytest.raises(RewardProcessingError):
        process_rewards(page)
    assert page.opened == []


def test_unconfirmed_end_of_list_is_not_reported_as_success():
    page = FakeRewardPage([])
    page.traversal_complete = lambda: False
    with pytest.raises(RewardProcessingError):
        process_rewards(page)
