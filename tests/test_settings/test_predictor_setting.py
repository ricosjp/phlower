import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from phlower.settings._predictor_setting import PhlowerPredictorSetting
from phlower.utils.enums import ModelSelectionType


@given(
    selection_type=st.sampled_from(ModelSelectionType),
)
def test__selection_mode_from_valid_names(selection_type: ModelSelectionType):
    target_epoch = (
        10 if selection_type == ModelSelectionType.SPECIFIED else None
    )
    setting = PhlowerPredictorSetting(
        selection_mode=selection_type.value, target_epoch=target_epoch
    )
    assert setting.selection_mode == selection_type.value


@pytest.mark.parametrize("mode_name", ["none", "my_method"])
def test__raise_error_not_existing_selection_mode(mode_name: str):
    with pytest.raises(ValueError) as ex:
        _ = PhlowerPredictorSetting(selection_mode=mode_name)
    assert f"{mode_name} selection mode does not exist" in str(ex.value)


@pytest.mark.parametrize("target_epoch", [0, 100, 20, 51])
@given(
    selection_type=st.sampled_from(ModelSelectionType),
)
def test__raise_error_target_epoch_is_set_when_not_specified(
    target_epoch: int, selection_type: ModelSelectionType
):
    assume(selection_type != ModelSelectionType.SPECIFIED)

    with pytest.raises(ValueError) as ex:
        _ = PhlowerPredictorSetting(
            selection_mode=selection_type.value, target_epoch=target_epoch
        )
    assert "target_epoch should be None" in str(ex.value)


@pytest.mark.parametrize("target_epoch", [-1, -100, None])
def test__raise_error_target_epoch_is_invalid_when_specified(target_epoch: int):
    with pytest.raises(ValueError) as ex:
        _ = PhlowerPredictorSetting(
            selection_mode="specified", target_epoch=target_epoch
        )
    assert "target_epoch should be non-negative value" in str(ex.value)
