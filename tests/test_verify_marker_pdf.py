import unittest

from scripts.marker_targets.verify_marker_pdf import parse_expected_ids, validate_detected_ids


class VerifyMarkerPdfTest(unittest.TestCase):
    def test_parse_expected_ids_rejects_duplicates_and_negative_values(self):
        self.assertEqual(parse_expected_ids("1, 2,3"), [1, 2, 3])
        with self.assertRaises(ValueError):
            parse_expected_ids("1,1")
        with self.assertRaises(ValueError):
            parse_expected_ids("-1")

    def test_validate_detected_ids_requires_each_expected_id_exactly_once(self):
        validate_detected_ids([1, 2, 3], [3, 1, 2])
        with self.assertRaises(ValueError):
            validate_detected_ids([1, 2, 3], [1, 2])
        with self.assertRaises(ValueError):
            validate_detected_ids([1, 2, 3], [1, 2, 3, 3])
        with self.assertRaises(ValueError):
            validate_detected_ids([1, 2, 3], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
