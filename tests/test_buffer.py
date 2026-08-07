import unittest

from chevron_agent import BoundedProvisionalBuffer, ProvisionalEntry


class BoundedProvisionalBufferTests(unittest.TestCase):
    def test_fifo_capacity_one_and_two(self) -> None:
        for capacity in (1, 2):
            buffer = BoundedProvisionalBuffer(capacity)
            evicted_ids = []
            for event_id in range(4):
                evicted = buffer.add(
                    ProvisionalEntry(event_id, event_id, {"value": event_id})
                )
                if evicted is not None:
                    evicted_ids.append(evicted.event_id)
            self.assertEqual(len(buffer), capacity)
            self.assertEqual(
                [entry.event_id for entry in buffer.entries],
                list(range(4 - capacity, 4)),
            )
            self.assertEqual(evicted_ids, list(range(4 - capacity)))

    def test_resolution_removes_only_matching_entry(self) -> None:
        buffer = BoundedProvisionalBuffer(2)
        buffer.add(ProvisionalEntry(10, 0, "a"))
        buffer.add(ProvisionalEntry(11, 1, "b"))
        resolved = buffer.resolve(10)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.payload, "a")
        self.assertIsNone(buffer.resolve(10))
        self.assertEqual([entry.event_id for entry in buffer.entries], [11])

    def test_invalid_capacity_and_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundedProvisionalBuffer(0)
        buffer = BoundedProvisionalBuffer(1)
        buffer.add(ProvisionalEntry(1, 0, None))
        with self.assertRaises(ValueError):
            buffer.add(ProvisionalEntry(1, 1, None))


if __name__ == "__main__":
    unittest.main()
