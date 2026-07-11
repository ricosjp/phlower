import pytest
import torch

from phlower.testing import assert_hash_equal


def test__assert_hash_equal_with_tensors():
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[1, 2], [3, 4]])
    assert_hash_equal(tensor1, tensor2)


def test__assert_hash_equal_with_different_tensors():
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6], [7, 8]])
    with pytest.raises(AssertionError):
        assert_hash_equal(tensor1, tensor2)


def test__assert_hash_equal_with_dicts():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 2, "a": 1}
    assert_hash_equal(dict1, dict2)


def test__assert_hash_equal_with_lists():
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    assert_hash_equal(list1, list2)


def test__assert_hash_equal_with_sets():
    set1 = {1, 2, 3}
    set2 = {3, 2, 1}
    assert_hash_equal(set1, set2)


def test__assert_hash_equal_with_mixed_types():
    mixed1 = {"a": [1, 2], "b": torch.tensor([3, 4])}
    mixed2 = {"b": torch.tensor([3, 4]), "a": [1, 2]}
    assert_hash_equal(mixed1, mixed2)


def test__assert_hash_equal_with_none():
    assert_hash_equal(None, None)


def test__assert_hash_equal_with_unsupported_type():
    class Unsupported:
        pass

    unsupported1 = Unsupported()
    unsupported2 = Unsupported()

    with pytest.raises(TypeError):
        assert_hash_equal(unsupported1, unsupported2)
