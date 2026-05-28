import torch
import torch.nn.functional as F
from ipdb import set_trace


def active_orthogonal_procrustes(global_emb, local_emb, active_ids, epsilon=1e-4, sample_ratio=0.4):
    total_indices = torch.arange(global_emb.size(0), device=global_emb.device)
    mask = torch.ones_like(total_indices, dtype=torch.bool)
    mask[active_ids] = False
    remaining_ids = total_indices[mask]
    num_noise_samples = int(len(remaining_ids) * sample_ratio)

    noise_ids = remaining_ids[torch.randperm(len(remaining_ids))[:num_noise_samples]]
    combined_ids = torch.cat([active_ids, noise_ids])
    A_sub = global_emb[combined_ids] 
    B_sub = local_emb[combined_ids]
    M = torch.matmul(B_sub.T, A_sub)
    M += epsilon * torch.eye(M.shape[0], device=M.device)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.T)
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = torch.matmul(U, V.T)
    return torch.matmul(local_emb, R)



def log_map(base, target):
        cos_theta = (base * target).sum(dim=-1, keepdim=True)
        cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        
        proj = target - cos_theta * base
        sin_theta = torch.sqrt(1 - cos_theta**2)

        factor = torch.where(
            sin_theta < 1e-6,
            torch.ones_like(sin_theta), 
            theta / sin_theta          
        )

        v = factor * proj
        return v

def exp_map(base, v):
    norm_v = v.norm(p=2, dim=-1, keepdim=True)
    
    mask = norm_v < 1e-6
    
    cos_norm = torch.cos(norm_v)
    sin_norm = torch.sin(norm_v)
    scale_factor = torch.where(mask, torch.ones_like(norm_v), sin_norm / norm_v)
    
    new_base = base * cos_norm + v * scale_factor
    
    return F.normalize(new_base, p=2, dim=-1)

