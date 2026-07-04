from temporal.cache import throttle_gamma


def test_monotonic_non_increasing():
    mags = [0, 0.5, 1, 2, 10]
    vals = [throttle_gamma(0.35, m) for m in mags]
    assert vals[0] == 0.35
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
