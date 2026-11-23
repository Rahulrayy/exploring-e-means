import math

# scaling laws from the paper (eq 6 and 7, coefficients from table 2, fitted on DEEP)


def pow2(x):
    # closest power of two, rounding in log space
    return 2 ** round(math.log2(x))


def alpha_law(k, B, n, T):
    return min(0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11, 1.0)


def knee_batch(k, n, T):
    B = 0.035 * k**0.13 * n**0.88 * T**0.81
    if n > 2e6:
        B = B / 8  # the paper uses knee/8 above 2M points
    return int(min(pow2(B), n))


def our_batch(k, n, T):
    # my batch rule: at least k so every cluster sees a few
    # points between revival checks, and around knee/8 so alpha_law stays below 1
    knee = knee_batch(k, n, T)
    return int(min(pow2(max(k, knee / 8)), knee, n))


def ladder(a_law, lo=-8, hi=6):
    # half-octave steps around the law's alpha, can't go above 1
    return sorted({min(a_law * 2 ** (j / 2), 1.0) for j in range(lo, hi + 1)})
