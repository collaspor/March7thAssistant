import pytest

from tasks.power.reward_relic_pagination import PaginationError, RewardPagination


def test_overlapping_rows_receive_stable_distinct_positions():
    pages = RewardPagination(columns=3)
    assert pages.observe(0, [3, 3], at_bottom=False) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert pages.observe(1, [3, 2], at_bottom=True) == [(2, 0), (2, 1)]
    assert pages.complete()


def test_identical_looking_rewards_remain_distinct_positions():
    pages = RewardPagination(columns=2)
    assert pages.observe(0, [2], at_bottom=True) == [(0, 0), (0, 1)]


def test_skip_in_rows_is_rejected():
    pages = RewardPagination(columns=2)
    pages.observe(0, [2], at_bottom=False)
    with pytest.raises(PaginationError):
        pages.observe(2, [2], at_bottom=True)


def test_inconsistent_overlap_is_rejected():
    pages = RewardPagination(columns=3)
    pages.observe(0, [3, 3], at_bottom=False)
    with pytest.raises(PaginationError):
        pages.observe(1, [2], at_bottom=True)


def test_unconfirmed_bottom_never_reports_complete():
    pages = RewardPagination(columns=3)
    pages.observe(0, [3], at_bottom=False)
    assert not pages.complete()


@pytest.mark.parametrize("start, counts, bottom", [(1, [2], True), (0, [0], True), (0, [3, 2], False), (None, [2], True)])
def test_untrustworthy_layout_is_rejected(start, counts, bottom):
    with pytest.raises(PaginationError):
        RewardPagination(columns=3).observe(start, counts, at_bottom=bottom)
