import torch
import numpy as np

def evaluate(args_dict, nets, stats, method):
    """
    Evaluate the policy network on your Zarr dataset.

    Returns:
        mean_cov: Mean squared error (MSE) across dataset
        mean_success: Placeholder (can be used for other metrics, here just 0)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets.eval()

    # Use the same dataset as during training
    from imle_policy.dataloaders.dataset_zarr import PolicyDataset
    dataset = PolicyDataset(
        dataset_path=args_dict['dataset_path'],
        pred_horizon=args_dict['pred_horizon'],
        obs_horizon=args_dict['obs_horizon'],
        action_horizon=args_dict['action_horizon'],
        dataset_percentage=0.1  # sample 10% for quick evaluation
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_dict['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            obs = batch['obs'].to(device)
            true_action = batch['action'].to(device)

            if method == 'rs_imle':
                noise = torch.randn(obs.shape[0], *true_action.shape[1:], device=device)
                pred_action = nets['policy_net'](obs, noise)
                # average over n_samples_per_condition
                pred_action = pred_action.mean(dim=1)
            else:
                # for diffusion or other methods, fallback to using obs as input
                pred_action = nets['policy_net'](obs, torch.randn_like(true_action))

            mse = torch.mean((pred_action - true_action) ** 2).item()
            total_mse += mse * obs.shape[0]
            total_samples += obs.shape[0]

    mean_mse = total_mse / total_samples
    mean_success = 0.0  # placeholder, can implement real success metric later

    return mean_mse, mean_success