import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    KL divergence loss with numerical stability guards.
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    Returns 0.0 if inputs contain NaN/inf or computation yields non-finite result.
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # If any input has NaN/inf, skip KL (return 0 to allow training to continue)
    if not (torch.isfinite(z_p).all() and torch.isfinite(logs_q).all() and
            torch.isfinite(m_p).all() and torch.isfinite(logs_p).all() and
            torch.isfinite(z_mask).all()):
        return torch.tensor(0.0, device=z_p.device, dtype=z_p.dtype)

    # Clamp log scale to prevent exp(-2*logs_p) overflow (NaN/inf)
    logs_p = torch.clamp(logs_p, min=-12.0, max=4.0)
    logs_q = torch.clamp(logs_q, min=-12.0, max=4.0)

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    mask_sum = torch.sum(z_mask)
    # Guard against empty mask (avoid div by zero)
    l = kl / torch.clamp(mask_sum, min=1e-6)
    # Clamp final loss to prevent NaN/inf propagation into optimizer
    l = torch.clamp(l, min=0.0, max=1e4)
    # Fallback: if still non-finite (e.g. from upstream), return 0
    if not torch.isfinite(l):
        return torch.tensor(0.0, device=z_p.device, dtype=z_p.dtype)
    return l
