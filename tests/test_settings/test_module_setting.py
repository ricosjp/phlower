from phlower.settings import ModuleSetting


def test_module_setting_checkpoint_defaults_and_roundtrips():
    default = ModuleSetting(nn_type="Identity", name="default")
    explicit = ModuleSetting(
        nn_type="Identity", name="explicit", checkpoint=True
    )

    assert default.checkpoint is False
    assert explicit.checkpoint is True
    assert (
        ModuleSetting.model_validate(default.model_dump()).checkpoint is False
    )
    assert (
        ModuleSetting.model_validate(explicit.model_dump()).checkpoint is True
    )


def test_module_setting_checkpoint_roundtrip_keeps_nested_parameters():
    original = ModuleSetting(
        nn_type="IsoGCN",
        name="graph",
        input_keys=["x", "neumann", "moment"],
        output_key="y",
        coeff=-1.7,
        nn_parameters={
            "nodes": [2, 2],
            "isoam_names": ["gx", "gy", "gz"],
            "propagations": ["convolution"],
            "self_network": {"activations": ["identity"], "bias": False},
            "neumann_setting": {
                "neumann_input_name": "neumann",
                "inversed_moment_name": "moment",
                "inversed_moment_source": "input_data",
            },
        },
    )
    restored = ModuleSetting.model_validate(
        {**original.model_dump(), "checkpoint": True}
    )

    assert original.checkpoint is False
    assert restored.checkpoint is True
    assert restored.model_dump(exclude={"checkpoint"}) == original.model_dump(
        exclude={"checkpoint"}
    )
