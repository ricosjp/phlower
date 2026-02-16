import pathlib

import jinja2

import phlower.nn._core_modules as _core_modules
import phlower.nn._core_preset_group_modules as _core_preset_group_modules


def _render_rst(template_name: str, output_name: str, **kwargs):
    template_path = (
        pathlib.Path(__file__).parent / f"_templates/{template_name}.rst.jinja2"
    )

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent))
    )
    template = env.get_template(template_path.name)

    rendered_str = template.render(**kwargs)
    output_path = pathlib.Path(__file__).parent / f"reference/{output_name}.rst"
    with open(output_path, "w") as f:
        f.write(rendered_str)


def render_nn_rst():
    _render_rst(
        template_name="nn",
        output_name="nn",
        module_names=[m.__name__ for m in _core_modules._all_models],
    )


def render_preset_rst():
    _render_rst(
        template_name="presets",
        output_name="presets",
        preset_names=[
            m.__name__ for m in _core_preset_group_modules._all_presets
        ],
    )


if __name__ == "__main__":
    render_nn_rst()
    render_preset_rst()
