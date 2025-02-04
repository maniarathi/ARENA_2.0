import math
from einops import rearrange, repeat, reduce
import torch as t


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(
    actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001
) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    return rearrange(t.arange(start=3, end=9), "(r c) -> r c", r=3)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


def rearrange_2() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return rearrange(t.arange(start=1, end=7), "(r c) -> r c", r=2)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


def rearrange_3() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    return rearrange(t.arange(1, 7), "(h w d) -> h w d", h=1, w=6)


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))


def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return reduce(temps, "(week day) -> week", "mean", day=7)


temps = t.Tensor(
    [71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83]
)
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)


def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    return temps - repeat(
        reduce(temps, "(week day) -> week", "mean", day=7),
        "h -> (h repeat)",
        repeat=7,
    )


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)


def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """
    weekly_average = reduce(temps, "(week day) -> week", "mean", day=7)
    broadcasted_average = repeat(weekly_average, "h -> (h repeat)", repeat=7)
    weekly_std = reduce(temps, "(week day) -> week", t.std, day=7)
    broadcasted_std = repeat(weekly_std, "h -> (h repeat)", repeat=7)
    return (temps - broadcasted_average) / broadcasted_std


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)
