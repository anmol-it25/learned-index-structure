from bisect import bisect_left

keys = [10, 20, 30, 40, 50]

def expected_search(key):
    pos = bisect_left(keys, key)
    return pos if pos < len(keys) and keys[pos] == key else -1

for key in [10, 30, 50, 25, 100]:
    print(key, expected_search(key))