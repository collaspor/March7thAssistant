"""结算奖励处理规则；界面适配器需提供经过确认的奖励身份和状态。"""


class RewardProcessingError(RuntimeError):
    """无法安全确认结算页状态时中止处理。"""


def process_rewards(page):
    """处理页面提供的奖励，返回新标记的数量。

    适配器必须为每件奖励提供跨滚动稳定且不同的身份，不能使用遗器名称或图像作为身份。
    不确定的识别结果必须报告为 unknown，且遍历失败时必须抛出异常。
    """
    seen = set()
    marked_count = 0
    for card in page.iter_cards():
        identity, kind, recommendation = card
        if identity is None:
            raise RewardProcessingError("无法确认奖励位置")
        if identity in seen:
            continue
        seen.add(identity)
        if kind == "other":
            continue
        if kind != "relic":
            raise RewardProcessingError("无法确认奖励类型")
        if recommendation == "present":
            continue
        if recommendation != "absent":
            raise RewardProcessingError("无法确认点赞标记")
        state = page.open_card(card)
        if state == "unmarked":
            if page.mark_discard(card) != "marked":
                raise RewardProcessingError("无法确认弃置标记")
            marked_count += 1
        elif state != "marked":
            raise RewardProcessingError("无法确认弃置状态")
        if not page.close_card(card):
            raise RewardProcessingError("无法返回结算页")
    if page.traversal_complete() is not True:
        raise RewardProcessingError("无法确认奖励列表已处理完毕")
    return marked_count
