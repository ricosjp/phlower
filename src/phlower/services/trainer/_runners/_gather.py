import torch.distributed as dist


def gather_losses_across_processes(
    local_losses: list[float],
) -> list[float]:
    rank = dist.get_rank()

    if rank == 0:
        gathered_losses = [None for _ in range(dist.get_world_size())]
    else:
        gathered_losses = None

    # NOTE: dist.gather_object is not suitable for large data
    # but it is acceptable here because the size of loss list is small.
    dist.gather_object(local_losses, gathered_losses, dst=0)

    if rank == 0:
        gathered_losses: list[list[float]]
        return sum(gathered_losses, start=[])
    else:
        return []


def gather_loss_details_across_processes(
    local_loss_details: list[dict[str, float]],
) -> list[dict[str, float]]:
    rank = dist.get_rank()

    if rank == 0:
        gathered_loss_details = [None for _ in range(dist.get_world_size())]
    else:
        gathered_loss_details = None

    # NOTE: dist.gather_object is not suitable for large data
    # but it is acceptable here because the size of loss list is small.
    dist.gather_object(local_loss_details, gathered_loss_details, dst=0)

    if rank == 0:
        gathered_loss_details: list[list[dict[str, float]]]
        return sum(gathered_loss_details, start=[])
    else:
        return []
