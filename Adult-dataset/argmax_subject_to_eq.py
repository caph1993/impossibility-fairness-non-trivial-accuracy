from typing import List


def argmax_st_zero(a: List[float], b: List[float]):
    '''
    Compute the list
        [r[i] for i in range(n)]
    that maximizes
        sum(a[i] * r[i] for i in range(n))
    subject to
        sum(b[i] * r[i] for i in range(n)) == 0
    and
        all(0 <= r[i] <= 1 for i in range(n))
    '''
    n = len(a)
    assert len(b) == n

    # Initialize x to zero. Set x[i]=1 greedily if b[i]==0 and a[i]>0
    x = [float(b[i] == 0 and a[i] > 0) for i in range(n)]

    # Separate indices depending on sign(b[i]), and sort using a[i] / abs(b[i])
    idx = [i for i in range(n) if b[i] != 0]
    idx.sort(key=lambda i: a[i] / abs(b[i]))
    pos = [i for i in idx if b[i] > 0]
    neg = [i for i in idx if b[i] < 0]

    while pos and neg:
        # Take the indices with maximal a[.]/abs(b[.])
        i, j = pos[-1], neg[-1]
        assert b[i] > 0 and b[j] < 0 and x[i] < 1 and x[j] < 1

        # Increase X[i] and X[j] greedily to maximum allowed
        ratio = -b[j] / b[i]
        new_i = x[i] + (1 - x[j]) * ratio
        new_j = x[j] + (1 - x[i]) / ratio
        new_i, new_j = (1, new_j) if new_j <= 1 else (new_i, 1)
        assert new_i <= 1 and new_j <= 1

        increment = (new_i - x[i]) * a[i] + (new_j - x[j]) * a[j]
        if increment <= 0:
            break

        x[i], x[j] = (new_i, new_j)
        if x[i] == 1:
            pos.pop()
        if x[j] == 1:
            neg.pop()
    return x