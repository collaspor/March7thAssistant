"""处理已确认的结算列表行号，不能代替滚动与画面识别。"""


class PaginationError(RuntimeError):
    """奖励列表位置或结束状态无法可靠确认。"""


class RewardPagination:
    def __init__(self, columns):
        if type(columns) is not int or columns <= 0:
            raise ValueError("列数必须为正整数")
        self.columns = columns
        self._rows = {}
        self._at_bottom = False

    def observe(self, first_row, row_counts, *, at_bottom):
        """记录已确认的完整行并返回之前未处理过的位置。

        调用方必须先独立确认首行的全局行号、每行卡片数和是否到达底部；
        无法确认时不得猜测，需直接停止。相同图案的不同卡片按位置区分。
        """
        if self._at_bottom:
            raise PaginationError("已到列表末尾，不能继续滚动")
        if type(first_row) is not int or first_row < 0 or type(at_bottom) is not bool:
            raise PaginationError("无法确认列表位置")
        if not isinstance(row_counts, (list, tuple)) or not row_counts:
            raise PaginationError("无法确认奖励行")
        if any(type(n) is not int or not 1 <= n <= self.columns for n in row_counts):
            raise PaginationError("奖励数量不合法")
        if any(n != self.columns for n in row_counts[:-1]) or (not at_bottom and row_counts[-1] != self.columns):
            raise PaginationError("存在未确认的部分行")
        if not self._rows and first_row != 0:
            raise PaginationError("未从列表顶部开始")
        if self._rows and first_row > max(self._rows):
            raise PaginationError("缺少已确认的重叠行")
        for offset, count in enumerate(row_counts):
            row = first_row + offset
            if row in self._rows and self._rows[row] != count:
                raise PaginationError("重叠行数量发生变化")
        last_row = first_row + len(row_counts) - 1
        if at_bottom and self._rows and last_row < max(self._rows):
            raise PaginationError("末尾位置与已观察的行不一致")
        new_positions = []
        for offset, count in enumerate(row_counts):
            row = first_row + offset
            if row not in self._rows:
                new_positions.extend((row, col) for col in range(count))
                self._rows[row] = count
        self._at_bottom = at_bottom
        return new_positions

    def complete(self):
        return self._at_bottom
