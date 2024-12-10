from __future__ import annotations

import dagstream

from phlower.settings._interface import (
    IModuleSetting,
    IReadOnlyReferenceGroupSetting,
)


class _SettingResolverAdapter:
    def __init__(self, setting: IModuleSetting) -> None:
        self._setting = setting

    @property
    def name(self) -> str:
        return self._setting.get_name()

    def __call__(
        self,
        *resolved_output: dict[str, int],
        parent: IReadOnlyReferenceGroupSetting | None = None,
    ) -> int:
        self._setting.resolve(*resolved_output, parent=parent)
        return self._setting.get_output_info()


def resolve_modules(
    starts: dict[str, int],
    modules: list[IModuleSetting],
    parent: IReadOnlyReferenceGroupSetting | None = None,
) -> list[dict[str, int]]:
    stream = dagstream.DagStream()
    resolvers = [_SettingResolverAdapter(layer) for layer in modules]
    name2node = {
        _resolver.name: stream.emplace(_resolver)[0] for _resolver in resolvers
    }

    for layer in modules:
        node = name2node[layer.get_name()]
        for to_name in layer.get_destinations():
            node.precede(name2node[to_name], pipe=True)

    _dag = stream.construct()

    executor = dagstream.StreamExecutor(_dag)
    results = executor.run(parent=parent, first_args=(starts,))

    return list(results.values())
