from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

@dataclass
class Node:
    leaf: bool
    keys: list = field(default_factory=list)
    children: list = field(default_factory=list)
    next: object = None

class BPlusTree:
    """In-memory B+ tree for unique integer keys; values are sorted ranks."""
    def __init__(self, order=32):
        if order < 3:
            raise ValueError('order must be >= 3')
        self.order = order
        self.root = Node(True)

    def insert(self, key, value):
        split = self._insert(self.root, key, value)
        if split is not None:
            sep, right = split
            self.root = Node(False, [sep], [self.root, right])

    def _insert(self, node, key, value):
        if node.leaf:
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.children[i] = value
                return None
            node.keys.insert(i, key)
            node.children.insert(i, value)
            if len(node.keys) <= self.order:
                return None
            mid = len(node.keys) // 2
            right = Node(True, node.keys[mid:], node.children[mid:], node.next)
            node.keys = node.keys[:mid]
            node.children = node.children[:mid]
            node.next = right
            return right.keys[0], right
        i = bisect_right(node.keys, key)
        split = self._insert(node.children[i], key, value)
        if split is not None:
            sep, right = split
            node.keys.insert(i, sep)
            node.children.insert(i + 1, right)
        if len(node.keys) <= self.order:
            return None
        mid = len(node.keys) // 2
        sep = node.keys[mid]
        right = Node(False, node.keys[mid + 1:], node.children[mid + 1:])
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return sep, right

    def search(self, key):
        node = self.root
        while not node.leaf:
            node = node.children[bisect_right(node.keys, key)]
        i = bisect_left(node.keys, key)
        return node.children[i] if i < len(node.keys) and node.keys[i] == key else -1

class LearnedIndex:
    """Linear least-squares rank predictor plus exact bisect correction."""
    def __init__(self, keys):
        self.keys = list(keys)
        self.fit()

    def fit(self):
        keys = self.keys
        n = len(keys)
        if n == 0:
            self.slope = self.intercept = 0.0
            self.origin = 0
            self.scale = 1
            self.max_error = 0.0
            return
        self.origin = keys[0]
        self.scale = max(keys[-1] - keys[0], 1)
        xs = [(x - self.origin) / self.scale for x in keys]
        mx = sum(xs) / n
        my = (n - 1) / 2
        var = sum((x - mx) ** 2 for x in xs)
        self.slope = sum((x - mx) * (i - my) for i, x in enumerate(xs)) / var if var else 0.0
        self.intercept = my - self.slope * mx
        self.max_error = max((abs(i - self.predict(x)) for i, x in enumerate(keys)), default=0.0)

    def predict(self, key):
        return self.slope * ((key - self.origin) / self.scale) + self.intercept

    def search(self, key):
        if not self.keys:
            return -1
        p = round(self.predict(key))
        radius = int(self.max_error) + 2
        lo = max(0, p - radius)
        hi = min(len(self.keys), p + radius + 1)
        i = bisect_left(self.keys, key, lo, hi)
        return i if i < len(self.keys) and self.keys[i] == key else -1

    def insert(self, key):
        i = bisect_left(self.keys, key)
        if i == len(self.keys) or self.keys[i] != key:
            self.keys.insert(i, key)
            self.fit()  # Refit on every insertion: deliberately simple baseline.
