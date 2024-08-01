:orphan:

{% extends "!autosummary/class.rst" %}

{#
Inherited methods from torch.nn.Module are not shown in this manual.
original templates are located in `.venv/lib/python3.10/site-packages/sphinx/ext/autosummary/templates/autosummary`
#}

{% block methods %}
   {% set showitems = ["from_setting", "get_nn_name", "forward"] %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   {% if item not in showitems %}
   {% else %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

{% endblock %}