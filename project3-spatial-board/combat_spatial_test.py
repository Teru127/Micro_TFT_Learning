import random
import unittest

from combat_spatial import position_factor, spatial_damage


class SpatialCombatTests(unittest.TestCase):
    def test_non_back_row_has_no_position_reduction(self):
        self.assertEqual(position_factor(0, random.Random(7)), 1.0)
        self.assertEqual(position_factor(1, random.Random(7)), 1.0)

    def test_back_row_damage_is_reduced(self):
        front_damage = spatial_damage(20, 10, 0, 1, random.Random(7))
        back_damage = spatial_damage(20, 10, 0, 2, random.Random(7))

        self.assertLess(back_damage, front_damage)

    def test_back_row_randomness_is_seeded(self):
        first = spatial_damage(20, 10, 0, 2, random.Random(23))
        second = spatial_damage(20, 10, 0, 2, random.Random(23))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
