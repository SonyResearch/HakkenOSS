from __future__ import annotations

import torch


@torch.no_grad()
def negative_sampler(
    pos_s: torch.Tensor,
    pos_r: torch.Tensor,
    pos_o: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    device: str | torch.device,
    negs_per_pos: int = 1,
    corrupt_probs: tuple[float, float, float] = (
        1 / 3,
        1 / 3,
        1 / 3,
    ),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate negative samples by corrupting positive triples.

    This function creates negative samples by randomly corrupting exactly one
    component (subject, relation, or object) of each positive triple according
    to the specified corruption probabilities.

    Args:
        pos_s (torch.Tensor): Positive subjects tensor.
        pos_r (torch.Tensor): Positive relations tensor.
        pos_o (torch.Tensor): Positive objects tensor.
        num_nodes (int): Total number of nodes in the graph.
        num_relations (int): Total number of relation types.
        device (torch.device): Device to perform computations on.
        negs_per_pos (int, optional): Number of negative samples to generate
            per positive sample. Defaults to 1.
        corrupt_probs (tuple[float, float, float], optional): Probabilities
            for corrupting (subject, relation, object) respectively.
            Must sum to 1.0. Defaults to (1/3, 1/3, 1/3).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - neg_s: Negative subjects tensor [num_pos * negs_per_pos].
            - neg_r: Negative relations tensor [num_pos * negs_per_pos].
            - neg_o: Negative objects tensor [num_pos * negs_per_pos].

    Note:
        The function ensures that corrupted components are different from
        their original values by using modular arithmetic when conflicts
        occur.
    """

    # Repeat positives to create num_samples negatives per positive
    if negs_per_pos > 1:
        pos_s = pos_s.repeat_interleave(negs_per_pos)
        pos_o = pos_o.repeat_interleave(negs_per_pos)
        pos_r = pos_r.repeat_interleave(negs_per_pos)

    num_samples = pos_r.numel()

    # Choose which component to corrupt for each sample
    # 0 -> corrupt subject, 1 -> corrupt relation, 2 -> corrupt object
    corrupt_type = torch.multinomial(
        torch.tensor(corrupt_probs, device=device),
        num_samples=num_samples,
        replacement=True,
    )

    neg_s = pos_s.clone()
    neg_r = pos_r.clone()
    neg_o = pos_o.clone()

    # Corrupt subjects
    mask_s = corrupt_type.eq(0)
    if mask_s.any():
        cand = torch.randint(0, num_nodes, (int(mask_s.sum().item()),), device=device)
        # avoid s' == original s
        same = cand.eq(pos_s[mask_s])
        if same.any():
            cand[same] = (cand[same] + 1) % num_nodes
        neg_s[mask_s] = cand

    # Corrupt relations
    mask_r = corrupt_type.eq(1)
    if mask_r.any():
        cand = torch.randint(0, num_relations, (int(mask_r.sum().item()),), device=device)
        same = cand.eq(pos_r[mask_r])
        if same.any():
            cand[same] = (cand[same] + 1) % num_relations
        neg_r[mask_r] = cand

    # Corrupt objects
    mask_o = corrupt_type.eq(2)
    if mask_o.any():
        cand = torch.randint(0, num_nodes, (int(mask_o.sum().item()),), device=device)
        same = cand.eq(pos_o[mask_o])
        if same.any():
            cand[same] = (cand[same] + 1) % num_nodes
        neg_o[mask_o] = cand

    return neg_s, neg_r, neg_o
