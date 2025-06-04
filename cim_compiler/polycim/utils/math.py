def get_factors(N):
    """
    get all factors of N
    """
    assert isinstance(N, int), f"N={N} must be an integer"
    assert N > 0, f"N={N} must be positive"

    factors = []
    for i in range(1, N + 1):
        if N % i == 0:
            factors.append(i)
    return factors


def factorize(N, T, depth=1, path=None, results=None):
    """
    factorize N into T factors
    """
    if path is None:
        path = []

    if results is None:
        results = []

    if T == 1:
        results.append(path + [N])
        return
    for i in get_factors(N):
        factorize(N // i, T - 1, i + 1, path + [i], results)
    return results
