import torch
from . import soft_dtw
from . import path_soft_dtw

def dilate_loss(outputs, targets, alpha, gamma, device):
    # Ensure outputs and targets are 3D: (batch_size, N_output, 1)
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(-1)  # Convert (batch_size, N_output) → (batch_size, N_output, 1)
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    batch_size, N_output = outputs.shape[0:2]
    softdtw_batch = soft_dtw.SoftDTWBatch.apply

    # Compute soft-DTW loss
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    # Compute temporal loss
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)

    Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(device)  # Fixed
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)

    # Final weighted loss
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal

    return loss  # Return a single tensor, fixing `.backward()` error
