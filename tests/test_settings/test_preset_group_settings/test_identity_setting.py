import pytest
from phlower.settings._preset_group_settings import IdentityPresetGroupSetting


@pytest.mark.parametrize(
    "input_to_output_map, input_keys, desired_msg",
    [
        (
            {"input1": "output1", "input2": "output2"},
            ["input1", "input2", "input3"],
            "Mapping for input key 'input3' is not found ",
        ),
        ({"a": "b"}, ["a", "c"], "Mapping for input key 'c' is not found "),
    ],
)
def test__detect_insufficient_inputs(
    input_to_output_map: dict[str, str], input_keys: list[str], desired_msg: str
):
    setting = IdentityPresetGroupSetting(
        input_to_output_map=input_to_output_map
    )

    class DummyModule:
        def get_name(self) -> str:
            return "DummyModule"

        def get_input_keys(self) -> list[str]:
            return input_keys

        def get_output_info(self) -> dict[str, int]:
            return {k: 1 for k in input_to_output_map.values()}

    dummy_module = DummyModule()

    with pytest.raises(ValueError, match=desired_msg):
        setting.confirm(dummy_module)


@pytest.mark.parametrize(
    "input_to_output_map, output_keys, desired_msg",
    [
        (
            {"input1": "output1", "input2": "output2"},
            ["output1", "output2", "output3"],
            "Mapping for output key 'output3' is not found ",
        ),
        ({"a": "b"}, ["b", "c"], "Mapping for output key 'c' is not found "),
    ],
)
def test__detect_insufficient_outputs(
    input_to_output_map: dict[str, str],
    output_keys: list[str],
    desired_msg: str,
):
    setting = IdentityPresetGroupSetting(
        input_to_output_map=input_to_output_map
    )

    class DummyModule:
        def get_name(self) -> str:
            return "DummyModule"

        def get_input_keys(self) -> list[str]:
            return list(input_to_output_map.keys())

        def get_output_info(self) -> dict[str, int]:
            return {k: 1 for k in output_keys}

    dummy_module = DummyModule()

    with pytest.raises(ValueError, match=desired_msg):
        setting.confirm(dummy_module)
