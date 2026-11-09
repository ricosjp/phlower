.. _usage:


Usage
=======


Installation
--------------

**phlower** can be installed with pip. Python 3.10 or newer is supported.

.. code-block:: bash

    pip install phlower


Activation checkpointing
------------------------

Set ``checkpoint: true`` on a module to trade backward recomputation for lower
activation memory. This option defaults to ``false``; existing configurations
keep their usual execution path. For example, a module entry can be written as:

.. code-block:: yaml

    nn_type: MLP
    name: encoder
    input_keys: [x]
    output_key: features
    checkpoint: true
    nn_parameters:
      nodes: [8, 32, 16]
      activations: [tanh, identity]

The initial supported module types are ``MLP``, ``GCN``, ``IsoGCN`` and
``EnEquivariantMLP``. Other module types and subclasses are rejected when
checkpointing is requested. Registered buffers are rejected when a checkpoint
would execute, since replay could update persistent state twice. Custom
stateful modifications to these modules are not supported.

Checkpointing is bypassed under ``torch.no_grad()`` or when the module has
``no_grad: true``. It uses PyTorch's non-reentrant implementation and preserves
the random-number-generator state, including for dropout. Phlower's forward
debug hooks run only on the original forward pass. Evaluation with gradients
enabled still uses checkpointing.

Inputs, field tensors and model parameters must remain valid and unchanged
until backward completes. Only the module's calculation is replayed; enclosing
groups and iteration solvers are not checkpointed by this option. Configuration
selection and memory/time measurements belong in the application benchmark.

This activation-memory option is separate from saving model checkpoints to
disk. See the `PyTorch checkpoint documentation
<https://docs.pytorch.org/docs/2.5/checkpoint.html>`_ for recomputation semantics.
