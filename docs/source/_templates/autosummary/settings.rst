:orphan:

{% extends "!autosummary/class.rst" %}

{#
Methods which startswith "model_" are not shown in this manual.
original templates are located in `.venv/lib/python3.10/site-packages/sphinx/ext/autosummary/templates/autosummary`
#}

{% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   {% if item.startswith("model_") %}
   {% else %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

{% endblock %}