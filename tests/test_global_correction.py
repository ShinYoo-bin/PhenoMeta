import numpy as np

from global_correction import combine_probabilities, select_alpha


def test_zero_alpha_preserves_attention_probability():
    attention = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    global_probability = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    np.testing.assert_allclose(
        combine_probabilities(attention, global_probability, 0.0), attention
    )


def test_equal_auroc_prefers_zero_alpha():
    labels = np.array([0, 0, 1, 1])
    attention = np.array([[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]])
    global_probability = np.full((4, 2), 0.5)
    selected = select_alpha(labels, attention, global_probability, [0.0, 0.05, 0.5])
    assert selected[3] == 0.0


def test_multiclass_combination_is_normalized():
    attention = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    global_probability = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    combined = combine_probabilities(attention, global_probability, 0.25)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    assert combined.shape == (2, 3)
