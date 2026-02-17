"""
.. _second:


Model Definition by Yaml file
----------------------------------------------------

Phlower offers a way to define models and its order by yaml file.

"""


###################################################################################################
# First of all, we would like to load sample yaml data. Please download sample sample yaml.
# `data.yml
# <https://github.com/ricosjp/phlower/blob/main/tutorials/basic_usages/sample_data/model/model.yml>`_
#
# we construct PhlowerSetting object from yaml file.

from phlower.settings import PhlowerSetting

setting = PhlowerSetting.read_yaml("sample_data/model/model.yml")


###################################################################################################
# Order of models must be DAG (Directed Acyclic Graph).
# To check such conditions, we call `resolve` function.

setting.model.network.resolve(is_first=True)


###################################################################################################
# In phlower, networks are packed into PhlowerGroupModule.
# PhlowerGroupModule is directly created from model setting.
#
# `draw` function generate a file following to mermaid format.

from phlower.nn import PhlowerGroupModule

model = PhlowerGroupModule.from_setting(setting.model.network)
model.draw("images")


###################################################################################################
# The output mermaid file is shown below.
#
# .. mermaid:: ../../tutorials/basic_usages/images/SAMPLE_MODEL.mmd
#
# According to this image, we can understand details of each model and data flow.
#
