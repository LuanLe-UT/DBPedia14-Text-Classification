import unittest

import numpy as np

from tfidf_svm import TfidfVectorizer, balanced_indices, evaluate_split


class TfidfVectorizerTests(unittest.TestCase):
    def test_builds_normalized_sparse_features_and_ignores_unknown_tokens(self):
        vectorizer = TfidfVectorizer(max_features=20, min_df=1)
        matrix = vectorizer.fit_transform(
            ["Cat cat sat", "Cat sat", "Dog slept"], batch_size=2
        )

        self.assertEqual(matrix.shape, (3, len(vectorizer.vocabulary_)))
        np.testing.assert_allclose(
            np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1,
            [1.0, 1.0, 1.0],
        )
        unknown = vectorizer.transform(["unknownword"], batch_size=1)
        self.assertEqual(unknown.nnz, 0)
        self.assertEqual(vectorizer.tokenize("A Cat, 42!"), ["a", "cat", "42"])

    def test_rejects_empty_fit_input(self):
        with self.assertRaisesRegex(ValueError, "empty document collection"):
            TfidfVectorizer().fit([])

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


if __name__ == "__main__":
    unittest.main()
