import pathlib

from phlower.nn._interface_module import IPhlowerModuleAdapter
from phlower.services.drawers._interface import IPhlowerDrawer


class MermaidDrawer(IPhlowerDrawer):
    def __init__(self) -> None:
        self._spliter = "\n" + " " * 4
        self._dir = "LR"

    def output(
        self,
        modules: list[IPhlowerModuleAdapter],
        file_path: pathlib.Path | str,
    ) -> None:
        """output content to file_path

        Parameters
        ----------
        graph : IDrawableGraph
            graph object to draw
        file_path : pathlib.Path
            path to file
        """
        context = self._generate(modules)
        with open(file_path, "w") as fw:
            fw.write(context)

    def _generate(self, modules: list[IPhlowerModuleAdapter]) -> str:
        context = ["stateDiagram", f"direction {self._dir}"]
        context.append("classDef alignCenter text-align:center")
        name2id: dict[str, str] = {}

        for i, module in enumerate(modules):
            name2id[module.name] = (state_name := f"state_{i}")
            _context = module.name
            if len(module.get_display_info()) != 0:
                _context += f"\n{module.get_display_info()}"
            context.append(f'state "{_context}" as {state_name}')

        for _, node in enumerate(modules):
            state_name = name2id[node.name]

            succesors = list(node.get_destinations())
            if len(succesors) == 0:
                context.append(f"{state_name} --> [*]")
                continue

            for successor in succesors:
                successor_state = name2id[successor]
                context.append(f"{state_name} --> {successor_state}: Pipe")

        # apply classDef
        all_state_names = ", ".join(list(name2id.values()))
        context.append(f"class {all_state_names} alignCenter")

        return self._spliter.join(context)
