from __future__ import annotations

import itertools

import numpy as np
import torch
from phlower_tensor import (
    ISimulationField,
    PhlowerTensor,
    SimulationField,
    phlower_tensor,
)
from phlower_tensor import functionals as functionals
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from scipy.spatial import KDTree

from phlower.nn._interface_module import (
    IPhlowerCorePresetGroupModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._preset_group_settings import (
    InterpolatorPresetGroupSetting,
)
from phlower.utils.exceptions import PhlowerInvalidArgumentsError


class InterpolatorPresetGroupModule(
    torch.nn.Module, IPhlowerCorePresetGroupModule
):
    """Semi-Lagrangian Attention is a neural network module for
    convection-dominated problems based on the Semi-Lagrangian scheme.

    Parameters
    ----------
    source_position_name: str
        Name to be used for source position data.
    target_position_name: str
        Name to be used for target position data.
    source_position_from_field: bool = True
        If False, take source position from tensors rather than field.
    target_position_from_field: bool = True
        If False, take target position from tensors rather than field.
    variable_names: list[str] | None = None
        Variable names to be interpolated. If not fed, all data are
        interpolated.
    source_field_name: str | None = None
        Name of the source to use for unbatching source fields.
    target_field_name: str | None = None
        Name of the target to use for unbatching source fields.
    n_head: int = 1
        The number of head for the multihead attention mechanism.
    weight_name: str | None = None
        Name to be used for weight data.
    length_scale: float = 1.0e-1
        Length scale to normalize distance-based attention.
    normalize_row: bool = True
        If True, normalize the attention matrix in the row direction.
    conservative: bool = False
        If True, normalize the attention matrix further to achieve conservation.
    bidirectional: bool = False
        If True, compute bidirectional interpolation assuming both source and
        target represent the same shape.
    k_nns: int = 9
        The number to be used for k nearest neighbor search.
        It should be larger than 1.
    d_nns: float | None = None
        If fed, perform nearest neighbor search based on the fed distance
        in addition to the k-NN search.
    learnable_length_scale: bool = False
        If True, learn length scale. In that case, the length_scale
        parameter above is used to initialize trainable weights.
    learnable_weight_transpose: bool = False
        If True, learn weight for the transposed attention matrix.
    learnable_weight_distance: bool = False
        If True, learn weight for the distance based attention matrix.
    recompute_distance_k_nns_if_grad_enabled: bool = False
        If True and torch.is_grad_enabled, re-compute distance for k-NN
        attention matrix to retain gradients.
    recompute_distance_d_nns_if_grad_enabled: bool = False
        If True and torch.is_grad_enabled, re-compute distance for distance
        attention matrix to retain gradient.
    unbatch: bool = True
        If True, unbatch the input data and field.
    epsilon: float = 1.0e-5
        Epsilon value to decimate small numbers.

    Examples
    --------
    >>> interpolator = InterpolatorPresetGroupModule(
    ...     source_position_name="source_position",
    ...     target_position_name="target_position",
    ... )
    >>> output = interpolator(data, field_data=field_data)
    """

    @classmethod
    def from_setting(
        cls, setting: InterpolatorPresetGroupSetting
    ) -> InterpolatorPresetGroupModule:
        """Create InterpolatorPresetGroupModule
        from InterpolatorPresetGroupSetting instance.

        Args:
            setting InterpolatorPresetGroupSetting): setting object

        Returns:
            InterpolatorPresetGroupModule: InterpolatorPresetGroupModule object
        """
        return InterpolatorPresetGroupModule(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "Interpolator"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        source_position_name: str,
        target_position_name: str,
        source_position_from_field: bool = True,
        target_position_from_field: bool = True,
        variable_names: list[str] | None = None,
        input_to_output_map: dict[str, str] | None = None,
        source_field_name: str | None = None,
        target_field_name: str | None = None,
        n_head: int = 1,
        weight_name: str | None = None,
        length_scale: float = 1.0e-1,
        normalize_row: bool = True,
        conservative: bool = False,
        bidirectional: bool = False,
        k_nns: int = 9,
        d_nns: float | None = None,
        learnable_length_scale: bool = False,
        learnable_weight_transpose: bool = False,
        learnable_weight_distance: bool = False,
        recompute_distance_k_nns_if_grad_enabled: bool = False,
        recompute_distance_d_nns_if_grad_enabled: bool = False,
        unbatch: bool = True,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()

        self._source_position_name = source_position_name
        self._target_position_name = target_position_name
        self._source_position_from_field = source_position_from_field
        self._target_position_from_field = target_position_from_field
        self._variable_names = variable_names
        self._input_to_output_map = input_to_output_map
        self._source_field_name = source_field_name
        self._target_field_name = target_field_name
        self._n_head = n_head
        self._weight_name = weight_name
        self._length_scale = length_scale
        self._normalize_row = normalize_row
        self._conservative = conservative
        self._bidirectional = bidirectional
        self._k_nns = k_nns
        if self._k_nns < 2:
            raise ValueError(f"Set k_nns > 1 (given: {self._k_nns})")
        self._d_nns = d_nns
        if self._d_nns is not None and self._d_nns <= 0:
            raise ValueError(f"Set d_nns > 0 (given: {self._d_nns})")

        self._learnable_length_scale = learnable_length_scale
        self._learnable_weight_transpose = learnable_weight_transpose
        self._learnable_weight_distance = learnable_weight_distance
        self._recompute_distance_k_nns_if_grad_enabled = (
            recompute_distance_k_nns_if_grad_enabled
        )
        self._recompute_distance_d_nns_if_grad_enabled = (
            recompute_distance_d_nns_if_grad_enabled
        )
        self._unbatch = unbatch
        self._epsilon = epsilon

        self._inv_length_scale = self._generate_head_parameter(
            learnable=self._learnable_length_scale,
            initial_value=1 / self._length_scale,
        )
        self._weight_head = self._generate_head_parameter(
            learnable=self._n_head > 1 and not self._conservative
        )
        self._weight_transpose = self._generate_head_parameter(
            learnable=self._learnable_weight_transpose
        )
        self._weight_distance = self._generate_head_parameter(
            learnable=self._learnable_weight_distance
        )

        self._validate()

    def _generate_head_parameter(
        self, learnable: bool = False, initial_value: float = 1.0
    ) -> torch.Tensor:
        if learnable:
            return torch.nn.Parameter(
                torch.rand(self._n_head) * initial_value,
                requires_grad=True,
            )

        return torch.tensor([initial_value] * self._n_head)

    def _validate(self):
        if self._conservative and not self._bidirectional:
            raise ValueError(
                "When conservative = True, set bidirectional = True"
            )
        if (
            not self._source_position_from_field
            and self._source_field_name is None
        ):
            raise ValueError(
                "When source_position_from_field is False, "
                "set source_field_name"
            )
        if (
            not self._target_position_from_field
            and self._target_field_name is None
        ):
            raise ValueError(
                "When target_position_from_field is False, "
                "set target_field_name"
            )

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def _get_weight(
        self,
        field_data: ISimulationField,
    ) -> PhlowerTensor | None:
        if self._weight_name is None:
            return None
        return field_data[self._weight_name]

    @property
    def source_field_name_for_unbatch(self) -> str:
        if self._source_position_from_field:
            return self._source_position_name
        return self._source_field_name

    @property
    def target_field_name_for_unbatch(self) -> str:
        if self._target_position_from_field:
            return self._target_position_name
        return self._target_field_name

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None,
        **kwards,
    ) -> IPhlowerTensorCollections:
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """
        if field_data is None:
            raise PhlowerInvalidArgumentsError("Field data is required")
        if self._variable_names is None:
            variable_names = list(data.keys())
        else:
            variable_names = self._variable_names
        if self._input_to_output_map is None:
            input_to_output_map = {v: v for v in variable_names}
        else:
            input_to_output_map = self._input_to_output_map

        if not self._unbatch:
            return phlower_tensor_collection(
                {
                    input_to_output_map[k]: v
                    for k, v in self._interpolate_multihead(
                        data,
                        field_data=field_data,
                        variable_names=variable_names,
                    ).items()
                }
            )

        n_nodes = field_data.get_batched_n_nodes(
            self.source_field_name_for_unbatch
        )
        unbatched_data = functionals.unbatch(data, n_nodes=n_nodes)
        unbatched_field = self._unbatch_field(field_data)

        unbatched_collections = [
            self._interpolate_multihead(
                d, field_data=f, variable_names=variable_names
            )
            for d, f in zip(unbatched_data, unbatched_field, strict=True)
        ]

        return phlower_tensor_collection(
            {
                input_to_output_map[name]: self._to_batch(
                    [collection[name] for collection in unbatched_collections],
                )
                for name in unbatched_collections[0].keys()
            }
        )

    def _to_batch(self, tensors: list[PhlowerTensor]) -> PhlowerTensor:
        if tensors[0].is_time_series:
            return functionals.to_batch(tensors, dense_concat_dim=1)[0]
        return functionals.to_batch(tensors, dense_concat_dim=0)[0]

    def _unbatch_field(
        self, field_data: ISimulationField
    ) -> list[ISimulationField]:
        n_source_nodes = field_data.get_batched_n_nodes(
            self.source_field_name_for_unbatch
        )
        n_target_nodes = field_data.get_batched_n_nodes(
            self.target_field_name_for_unbatch
        )
        if n_source_nodes == n_target_nodes:
            return functionals.unbatch(field_data, n_nodes=n_source_nodes)

        unbatched_dict_source = functionals.unbatch(
            phlower_tensor_collection(
                {
                    k: v
                    for k, v in field_data.items()
                    if v.n_vertices() == sum(n_source_nodes)
                }
            ),
            n_nodes=n_source_nodes,
        )
        unbatched_dict_target = functionals.unbatch(
            phlower_tensor_collection(
                {
                    k: v
                    for k, v in field_data.items()
                    if v.n_vertices() == sum(n_target_nodes)
                }
            ),
            n_nodes=n_target_nodes,
        )
        for i_batch, dict_source in enumerate(unbatched_dict_source):
            dict_source.update(unbatched_dict_target[i_batch])
        return [
            SimulationField(dict_source, batch_info=None)
            for dict_source in unbatched_dict_source
        ]

    def _interpolate_multihead(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None,
        variable_names: str,
        **kwards,
    ) -> IPhlowerTensorCollections:
        if self._source_position_from_field:
            source_position = field_data[self._source_position_name]
        else:
            source_position = data.pop(self._source_position_name)
        if self._target_position_from_field:
            target_position = field_data[self._target_position_name]
        else:
            target_position = data.pop(self._target_position_name)
        weight = self._get_weight(field_data)

        multihead_variables = []

        for i_head in range(self._n_head):
            head_variables = phlower_tensor_collection(
                {
                    k: self._access_head(v, i_head)
                    for k, v in data.items()
                    if k in variable_names
                }
            )
            if source_position.shape[-1] == 1:
                head_source_position = source_position
            else:
                head_source_position = source_position[..., i_head : i_head + 1]
            if target_position.shape[-1] == 1:
                head_target_position = target_position
            else:
                head_target_position = target_position[..., i_head : i_head + 1]

            tmp_variable = (
                self._interpolated_signle_head(
                    head_variables,
                    head_source_position,
                    head_target_position,
                    weight=weight,
                    inv_length_scale=self._inv_length_scale[i_head],
                    weight_transpose=self._weight_transpose[i_head],
                    weight_distance=self._weight_distance[i_head],
                )
                * self._weight_head[i_head]
            )
            multihead_variables.append(tmp_variable)

        ret_variable = phlower_tensor_collection(
            {
                variable_name: torch.cat(
                    [
                        head_variables[variable_name]
                        for head_variables in multihead_variables
                    ],
                    dim=-1,
                )
                for variable_name in data.keys()
            }
        )
        return ret_variable

    def _access_head(self, tensor: PhlowerTensor, i_head: int) -> PhlowerTensor:
        n_feat_head, mod = divmod(tensor.shape[-1], self._n_head)
        if mod != 0:
            raise ValueError(
                f"Invalid tensor shape {tensor.shape} for head{self._n_head}"
            )
        return tensor[..., i_head * n_feat_head : (i_head + 1) * n_feat_head]

    def _interpolated_signle_head(
        self,
        variables: IPhlowerTensorCollections,
        source_position: PhlowerTensor,
        target_position: PhlowerTensor,
        weight: PhlowerTensor,
        inv_length_scale: torch.Tensor,
        weight_transpose: torch.Tensor,
        weight_distance: torch.Tensor,
    ) -> IPhlowerTensorCollections:
        source_tree = KDTree(np.squeeze(source_position.numpy()))
        target_tree = KDTree(np.squeeze(target_position.numpy()))

        # Shape: [n_target, n_source]
        sparse, _ = self._compute_sparse(
            base_position=source_position,
            query_position=target_position,
            inv_length_scale=inv_length_scale,
            tree=source_tree,
            weight=weight,
        )

        if self._bidirectional:
            # NOTE: For conservation, need to perform NNS in both direction and
            # transpose one of them to be sure that no empty row / col exists

            # Shape: [n_source, n_target]
            sparse_near_source, _ = self._compute_sparse(
                base_position=target_position,
                query_position=source_position,
                inv_length_scale=inv_length_scale,
                tree=target_tree,
                weight=weight,
            )
            sparse += sparse_near_source.transpose(0, 1) * (
                torch.abs(weight_transpose)
            )

        if self._d_nns is not None:
            # Shape: [n_target, n_source]
            sparse_dist_near_source, _ = self._compute_sparse_dist(
                base_position=target_position,
                query_position=source_position,
                inv_length_scale=inv_length_scale,
                base_tree=target_tree,
                query_tree=source_tree,
                weight=weight,
            )
            sparse += (
                sparse_dist_near_source
                * torch.abs(weight_distance)
                * torch.abs(weight_transpose)
            )

        if self._d_nns is not None and self._bidirectional:
            # Shape: [n_source, n_target]
            sparse_dist, _ = self._compute_sparse_dist(
                base_position=source_position,
                query_position=target_position,
                inv_length_scale=inv_length_scale,
                base_tree=source_tree,
                query_tree=target_tree,
                weight=weight,
            )
            sparse += sparse_dist.transpose(0, 1) * torch.abs(weight_distance)

        if self._conservative:
            ratio = sparse.shape[0] / sparse.shape[1]
        else:
            ratio = 1.0

        interpolated_variables = phlower_tensor_collection(
            {
                k: self._compute_normalized_spmm(sparse, v) * ratio
                for k, v in variables.items()
            }
        )
        return interpolated_variables

    def _compute_normalized_spmm(
        self,
        sparse: PhlowerTensor,
        variable: PhlowerTensor,
    ) -> PhlowerTensor:
        is_time_series = variable.is_time_series
        if is_time_series:
            str_t = "t"
        else:
            str_t = ""

        # Compute normalizer
        if self._normalize_row:
            weight_row = phlower_tensor(
                1 / sparse.to_tensor().sum(dim=1).detach().to_dense(),
                is_time_series=False,
                is_voxel=False,
                dimension=sparse.dimension,
            )
        else:
            weight_row = phlower_tensor(
                torch.ones(sparse.shape[0], device=sparse.device),
                dimension=sparse.dimension,
            )

        if self._conservative:
            mt_wrow = functionals.spmm(
                sparse.transpose(0, 1), weight_row[..., None]
            )[..., 0]
            weight_col_data = 1 / mt_wrow.detach().to_tensor().to_dense()
        else:
            weight_col_data = torch.ones(variable.n_vertices()).to(
                variable.device
            )

        weight_col = phlower_tensor(
            weight_col_data,
            is_time_series=False,
            is_voxel=False,
            dimension=sparse.dimension,
        )

        # Compute [ret]_i = [weight_row]_i [sparse]_{ij} [weight_col]_j phi_j
        interpolated_variable = functionals.einsum(
            f"i,{str_t}i...->{str_t}i...",
            weight_row,
            functionals.spmm(
                sparse,
                functionals.einsum(
                    f"j,{str_t}j...->{str_t}j...",
                    weight_col,
                    variable,
                    is_time_series=is_time_series,
                    is_voxel=False,
                    dimension=variable.dimension,
                ),
            ),
            is_time_series=is_time_series,
            is_voxel=False,
            dimension=variable.dimension,
        )
        if torch.any(torch.isnan(interpolated_variable.to_tensor())):
            raise ValueError(
                "NaN detected in the forward "
                "of InterpolatorPresetGroupModule.\n"
                f"sparse data:\n{sparse.to_tensor().coalesce().values()}\n"
                f"ret:\n{interpolated_variable}\n"
            )

        return interpolated_variable

    def _compute_sparse(
        self,
        base_position: PhlowerTensor,
        query_position: PhlowerTensor,
        inv_length_scale: PhlowerTensor | float,
        tree: KDTree | None = None,
        weight: PhlowerTensor | None = None,
    ) -> tuple[PhlowerTensor, PhlowerTensor]:
        if tree is None:
            tree = KDTree(np.squeeze(base_position.numpy()))
        _np_dist_near_query, indices_near_query = tree.query(
            np.squeeze(query_position.numpy()),
            k=self._k_nns,
        )

        if (
            self._recompute_distance_k_nns_if_grad_enabled
            and torch.is_grad_enabled()
        ):
            # Re-compute distance in PhlowerTensor to enable autograd
            dist_near_query = functionals.squeeze(
                torch.linalg.norm(
                    base_position[indices_near_query]
                    - query_position[:, None, :],
                    axis=-2,
                )
            ).to_tensor()
            weight_near_query = torch.exp(-dist_near_query * inv_length_scale)
        else:
            # Skip distance re-computation if no grad is required
            dist_near_query = torch.tensor(_np_dist_near_query).to(
                base_position.device,
                base_position.dtype,
            )
            weight_near_query = torch.exp(-dist_near_query * inv_length_scale)

        if weight is not None:
            weight_near_query = (
                weight_near_query
                * functionals.squeeze(weight)[indices_near_query].to_tensor()
            )

        # NOTE: Manage too small weight for stable computation
        weight_near_query = torch.clamp(weight_near_query, min=self._epsilon)

        n = len(base_position)
        m = len(query_position)

        row = torch.from_numpy(np.repeat(np.arange(m), self._k_nns)).to(
            weight_near_query.device
        )
        col = torch.flatten(torch.from_numpy(indices_near_query)).to(
            weight_near_query.device
        )

        sparse_near_query = torch.sparse_coo_tensor(
            indices=torch.stack([row, col], dim=0),
            values=torch.flatten(weight_near_query),
            size=(m, n),
        )

        if base_position.has_dimension:
            dimension = {}
        else:
            dimension = None
        return (
            phlower_tensor(sparse_near_query, dimension=dimension),
            dist_near_query,
        )

    def _compute_sparse_dist(
        self,
        base_position: PhlowerTensor,
        query_position: PhlowerTensor,
        inv_length_scale: PhlowerTensor | float,
        base_tree: KDTree | None = None,
        query_tree: KDTree | None = None,
        weight: PhlowerTensor | None = None,
    ) -> tuple[PhlowerTensor, PhlowerTensor]:
        if base_tree is None:
            base_tree = KDTree(np.squeeze(base_position.numpy()))
        if query_tree is None:
            query_tree = KDTree(np.squeeze(query_position.numpy()))

        if (
            self._recompute_distance_d_nns_if_grad_enabled
            and torch.is_grad_enabled()
        ):
            indices = base_tree.query_ball_tree(query_tree, r=self._d_nns)
            distances = torch.cat(
                [
                    torch.linalg.norm(
                        base_position[i_row] - query_position[cols],
                        dim=-2,
                    )
                    for i_row, cols in enumerate(indices)
                ],
                dim=0,
            )

            # NOTE: No need to overwrite weights because the max distance is
            #       bounded
            weight_distance = torch.exp(
                -distances.to_tensor() * inv_length_scale
            )
            row = torch.tensor(
                list(
                    itertools.chain.from_iterable(
                        [
                            [i_row] * len(cols)
                            for i_row, cols in enumerate(indices)
                        ]
                    )
                )
            ).to(weight_distance.device)
            col = torch.tensor(list(itertools.chain.from_iterable(indices))).to(
                weight_distance.device
            )

        else:
            scipy_sparse = base_tree.sparse_distance_matrix(
                query_tree,
                self._d_nns,
                output_type="coo_matrix",
            )
            distances = torch.from_numpy(scipy_sparse.data).to(
                base_position.device,
                base_position.dtype,
            )[:, None]
            weight_distance = torch.exp(-distances * inv_length_scale)
            row = torch.tensor(scipy_sparse.row).to(distances.device)
            col = torch.tensor(scipy_sparse.col).to(distances.device)

        if weight is not None and len(weight_distance) > 0:
            weight_distance = (
                weight_distance
                * functionals.squeeze(weight)[col, None].to_tensor()
            )

        n = len(base_position)
        m = len(query_position)
        sparse = torch.sparse_coo_tensor(
            indices=torch.stack([row, col], dim=0),
            values=torch.flatten(weight_distance),
            size=(n, m),
        )

        if base_position.has_dimension:
            dimension = {}
        else:
            dimension = None
        return (
            phlower_tensor(sparse, dimension=dimension),
            distances,
        )
