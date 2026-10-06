"""
Lab 7: The Collision Resolver -- starter.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert or update. Resize once load factor exceeds 0.75."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]

        # Update existing key.
        while current is not None:
            if current.key == key:
                current.value = value
                return
            current = current.next

        # Add a new node to the front of the bucket.
        new_node = _ChainNode(key, value)
        new_node.next = self._buckets[index]
        self._buckets[index] = new_node
        self._count += 1

        # Resize after insertion if load factor exceeds 0.75.
        if self._count / len(self._buckets) > 0.75:
            self._resize(len(self._buckets) * 2)

    def get(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]

        while current is not None:
            if current.key == key:
                return current.value
            current = current.next

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove key. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]
        previous: Optional[_ChainNode[K, V]] = None

        while current is not None:
            if current.key == key:
                if previous is None:
                    self._buckets[index] = current.next
                else:
                    previous.next = current.next

                self._count -= 1
                return

            previous = current
            current = current.next

        raise KeyError(key)

    def _resize(self, new_size: int) -> None:
        """Create a larger bucket array and rehash all nodes."""
        old_buckets = self._buckets
        self._buckets = [None] * new_size

        for node in old_buckets:
            current = node

            while current is not None:
                next_node = current.next

                index = hash(current.key) % new_size
                current.next = self._buckets[index]
                self._buckets[index] = current

                current = next_node


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert or update. Resize once load factor exceeds 0.7."""

        # Resize before insertion if this insertion would exceed 0.7.
        if (self._count + 1) / len(self._keys) > 0.7:
            self._resize(len(self._keys) * 2)

        index = hash(key) % len(self._keys)
        first_tombstone = -1

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                # Reuse the first tombstone encountered.
                if first_tombstone != -1:
                    index = first_tombstone

                self._keys[index] = key
                self._values[index] = value
                self._count += 1
                return

            if current_key is _TOMBSTONE:
                if first_tombstone == -1:
                    first_tombstone = index

            elif current_key == key:
                # Update existing value.
                self._values[index] = value
                return

            index = (index + 1) % len(self._keys)

        # This should not normally happen because of the load-factor rule.
        self._resize(len(self._keys) * 2)
        self.insert(key, value)

    def search(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        index = hash(key) % len(self._keys)

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                value = self._values[index]

                # A real entry always has a value slot, even if V itself
                # is Optional.
                return value  # type: ignore

            index = (index + 1) % len(self._keys)

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove key using a tombstone."""
        index = hash(key) % len(self._keys)

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                self._keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return

            index = (index + 1) % len(self._keys)

        raise KeyError(key)

    def _resize(self, new_size: int) -> None:
        """Resize and rehash all live entries."""
        old_keys = self._keys
        old_values = self._values

        self._keys = [None] * new_size
        self._values = [None] * new_size
        old_count = self._count
        self._count = 0

        for i, key in enumerate(old_keys):
            if key is not None and key is not _TOMBSTONE:
                self._insert_without_resize(
                    key, old_values[i]  # type: ignore
                )

        assert self._count == old_count

    def _insert_without_resize(self, key: K, value: V) -> None:
        """Insert a known-new key without checking the load factor."""
        index = hash(key) % len(self._keys)

        while self._keys[index] is not None:
            index = (index + 1) % len(self._keys)

        self._keys[index] = key
        self._values[index] = value
        self._count += 1


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.

    Uses prime table sizes to avoid the probing-cycle problem.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert or update. Resize before load factor exceeds 0.7."""

        if (self._count + 1) / len(self._keys) > 0.7:
            self._resize(self._next_prime(len(self._keys) * 2))

        start = hash(key) % len(self._keys)
        first_tombstone = -1

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                if first_tombstone != -1:
                    index = first_tombstone

                self._keys[index] = key
                self._values[index] = value
                self._count += 1
                return

            if current_key is _TOMBSTONE:
                if first_tombstone == -1:
                    first_tombstone = index

            elif current_key == key:
                self._values[index] = value
                return

        # Should only be reached in an unusual probing situation.
        self._resize(self._next_prime(len(self._keys) * 2))
        self.insert(key, value)

    def search(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        start = hash(key) % len(self._keys)

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                value = self._values[index]
                return value  # type: ignore

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove key using a tombstone."""
        start = hash(key) % len(self._keys)

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                self._keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return

        raise KeyError(key)

    def _resize(self, new_size: int) -> None:
        """Resize and rehash all live entries."""
        old_keys = self._keys
        old_values = self._values

        self._keys = [None] * new_size
        self._values = [None] * new_size
        old_count = self._count
        self._count = 0

        for i, key in enumerate(old_keys):
            if key is not None and key is not _TOMBSTONE:
                self._insert_without_resize(
                    key, old_values[i]  # type: ignore
                )

        assert self._count == old_count

    def _insert_without_resize(self, key: K, value: V) -> None:
        """Insert a known-new key without resizing."""
        start = hash(key) % len(self._keys)

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)

            if self._keys[index] is None:
                self._keys[index] = key
                self._values[index] = value
                self._count += 1
                return

        raise RuntimeError("Quadratic probing could not find an empty slot.")

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Return True if n is prime."""
        if n < 2:
            return False

        if n == 2:
            return True

        if n % 2 == 0:
            return False

        divisor = 3

        while divisor * divisor <= n:
            if n % divisor == 0:
                return False
            divisor += 2

        return True

    @classmethod
    def _next_prime(cls, n: int) -> int:
        """Return the smallest prime >= n."""
        while not cls._is_prime(n):
            n += 1

        return n