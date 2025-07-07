import numpy as np
import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
import json


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", window_lens=[1, 1], guide_w=0, save_attn=False, save_token=False):
    model.load_state_dict(torch.load(foldername + "/model.pth"))
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        nmse_total = 0
        nmae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_tt_attns = []
        all_tf_attns = []
        all_tokens = []
        with tqdm(test_loader, mininterval=1.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample, guide_w)

                if save_attn:
                    if save_token:
                        samples, c_target, eval_points, observed_points, observed_time, attns, tokens = output
                    else:
                        samples, c_target, eval_points, observed_points, observed_time, attns = output
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                if save_attn:
                    f = lambda x: x.detach().mean(dim=1).unsqueeze(1)
                    attns = [(f(attn1), f(attn2)) for attn1, attn2 in attns] 
                    tt_attns, tf_attns = zip(*attns)
                    tt_attns = torch.cat(tt_attns, 1)
                    tf_attns = torch.cat(tf_attns, 1)
                    tt_attns = tt_attns.chunk(2, dim=0)[0]
                    tf_attns = tf_attns.chunk(2, dim=0)[0]
                    all_tt_attns.append(tt_attns) 
                    all_tf_attns.append(tf_attns) 
                if save_token:
                    all_tokens.extend(tokens)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                nmse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                )
                nmae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                )

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                nmse_total += nmse_current.sum().item()
                nmae_total += nmae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "nmse_total": nmse_total / evalpoints_total,
                        "nmae_total": nmae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            # if save_attn:
            #     all_tt_attns = torch.cat(all_tt_attns, dim=0)
            #     all_tf_attns = torch.cat(all_tf_attns, dim=0)


            # np.save(foldername + "/generated_nsample" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_generated_samples.cpu().numpy())
            # np.save(foldername + "/target_" + str(nsample) + "_guide" + str(guide_w) + ".npy", all_target.cpu().numpy())
            # if save_attn:
            #     np.save(foldername + "/all_tt_attns" + ".npy", all_tt_attns.cpu().numpy())
            #     np.save(foldername + "/all_tf_attns" + ".npy", all_tf_attns.cpu().numpy())
            # if save_token:
            #     np.save(foldername + "/tokens" + ".npy", np.asarray(all_tokens))

            results = {
                "guide_w": guide_w,
                "MSE": nmse_total / evalpoints_total,
                "MAE": nmae_total / evalpoints_total,
            }
            with open(foldername + "config_results.json", "a") as f:
                json.dump(results, f, indent=4)
            print("MSE:", nmse_total / evalpoints_total)
            print("MAE:", nmae_total / evalpoints_total)
    return nmse_total / evalpoints_total
