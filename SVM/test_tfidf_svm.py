import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_svm import balanced_indices, evaluate_split


class SvmUtilityTests(unittest.TestCase):
    def test_reports_accuracy_and_macro_metrics(self):
        result = evaluate_split(np.asarray([0, 1, 1]), np.asarray([0, 1, 0]))

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertIn("precision_macro", result)
        self.assertIn("recall_macro", result)
        self.assertIn("f1_macro", result)

    def test_limited_dataset_sampling_covers_classes(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        selected = balanced_indices(labels, 5)

        self.assertEqual(len(selected), 5)
        self.assertEqual([labels[index] for index in selected], [0, 0, 1, 1, 2])

    def test_sklearn_vectorizer_uses_configured_token_pattern_and_l2_norm(self):
        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[^\W_]+\b",
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm="l2",
            min_df=1,
        )
        matrix = vectorizer.fit_transform(["Cat cat sat", "Cat sat", "Dog slept"])

        self.assertIn("cat", vectorizer.vocabulary_)
        self.assertEqual(matrix.shape[0], 3)
        np.testing.assert_allclose(
            np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1,
            [1.0, 1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
