import pytest
import torch
from merge.pipeline import ModelMerger


@pytest.fixture
def base_model_dict():
    return {
        "model.layers.0.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "model.layers.0.bias": torch.tensor([0.1, 0.2]),
        "model.layers.1.weight": torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        "model.embed_tokens.position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }


@pytest.fixture
def target_model_dicts():
    target_1 = {
        "model.layers.0.weight": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "model.layers.0.bias": torch.tensor([0.5, 0.6]),
        "model.layers.1.weight": torch.tensor([[3.0, 3.0], [3.0, 3.0]]),
        "model.embed_tokens.position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    target_2 = {
        "model.layers.0.weight": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
        "model.layers.0.bias": torch.tensor([0.2, 0.4]),
        "model.layers.1.weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "model.embed_tokens.position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    return [target_1, target_2]


def test_model_merger_slerp(base_model_dict, target_model_dicts):
    merger = ModelMerger()
    merged, summary = merger.merge_state_dicts(
        method="slerp",
        base_state_dict=base_model_dict,
        target_state_dicts=[target_model_dicts[0]],
        slerp_t=0.5,
    )
    
    assert summary["method"] == "slerp"
    assert summary["merged_parameters"] > 0
    # Non-float buffer preserved
    assert torch.equal(merged["model.embed_tokens.position_ids"], base_model_dict["model.embed_tokens.position_ids"])
    assert merged["model.embed_tokens.position_ids"].dtype == torch.long


def test_model_merger_ties(base_model_dict, target_model_dicts):
    merger = ModelMerger()
    merged, summary = merger.merge_state_dicts(
        method="ties",
        base_state_dict=base_model_dict,
        target_state_dicts=target_model_dicts,
        ties_top_k=0.5,
    )
    
    assert summary["method"] == "ties"
    assert summary["num_targets"] == 2
    assert "model.layers.0.weight" in merged
    assert merged["model.layers.0.weight"].shape == (2, 2)


def test_model_merger_dare(base_model_dict, target_model_dicts):
    merger = ModelMerger()
    merged, summary = merger.merge_state_dicts(
        method="dare",
        base_state_dict=base_model_dict,
        target_state_dicts=target_model_dicts,
        dare_drop_rate=0.2,
    )
    
    assert summary["method"] == "dare"
    assert "model.layers.1.weight" in merged


def test_model_merger_filter_regex(base_model_dict, target_model_dicts):
    merger = ModelMerger()
    # Only merge layer 0, leaving layer 1 as base
    merged, summary = merger.merge_state_dicts(
        method="slerp",
        base_state_dict=base_model_dict,
        target_state_dicts=[target_model_dicts[0]],
        filter_regex=r"layers\.0\.",
    )
    
    # Layer 1 must remain strictly identical to base
    assert torch.equal(merged["model.layers.1.weight"], base_model_dict["model.layers.1.weight"])
    assert summary["skipped_filtered_parameters"] > 0


def test_model_merger_missing_keys_raises(base_model_dict):
    merger = ModelMerger()
    invalid_target = {"model.layers.0.weight": torch.randn(2, 2)}
    with pytest.raises(KeyError):
        merger.merge_state_dicts(
            method="slerp",
            base_state_dict=base_model_dict,
            target_state_dicts=[invalid_target],
        )


def test_model_merger_invalid_method(base_model_dict, target_model_dicts):
    merger = ModelMerger()
    with pytest.raises(ValueError):
        merger.merge_state_dicts(
            method="unknown_algo",
            base_state_dict=base_model_dict,
            target_state_dicts=target_model_dicts,
        )
