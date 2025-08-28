import pydantic


class SlidingWindow(pydantic.BaseModel):
    offset: int
    """
    Offset of the sliding window.
    """

    size: int
    """
    Step size of the sliding window.
    """

    stride: int
    """
    Stride of the sliding window.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    def get_total_items(self, time_length: int) -> int:
        if time_length < self.offset + self.size:
            return 0

        # NOTE: drop the last incomplete window
        return 1 + (time_length - self.offset - self.size) // self.stride

    def get_slice(self, index: int) -> slice:
        start = self.offset + index * self.stride
        return slice(start, start + self.size)
