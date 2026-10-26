# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import traceback
import hydra
import numpy as np
import termcolor

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from adjoint_samplers.components.mode_correction import buffer_correction_diagnostic
from adjoint_samplers.components.sde import ControlledSDE, sdeint, enable_cuda_graph_sampling
from adjoint_samplers.train_loop import train_one_epoch
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode
from adjoint_samplers.utils.dist_utils import GMM
from omegaconf import OmegaConf

# `${gmm_std:${dim},c,${radius}}` = c x per-coordinate std of the GMM target, to size sigma_max / scale
OmegaConf.register_new_resolver(
    "gmm_std", lambda dim, factor=1, radius=None: factor * GMM.coord_std(dim, radius), replace=True,
)
def _hetero_std(dim, K, seed, radius_min, radius_max, scale_min, scale_max, weight_ratio, mult, min_sep=3.5,
                layout="random", simplex_radius=4.25):
    from adjoint_samplers.utils.dist_utils import HeteroGMM
    g = HeteroGMM(dim=dim, K=K, seed=seed, radius_min=radius_min, radius_max=radius_max, scale_min=scale_min,
                  scale_max=scale_max, weight_ratio=weight_ratio, min_sep=min_sep, layout=layout, simplex_radius=simplex_radius)
    return mult * g.coord_std()


OmegaConf.register_new_resolver("hetero_std", _hetero_std, replace=True)


def _manywell_std(dim, mult, well_a=6.0, well_b=0.5):
    from adjoint_samplers.utils.dist_utils import ManyWell
    return mult * ManyWell(dim=dim, a=-well_b, b=-well_a).coord_std()


OmegaConf.register_new_resolver("manywell_std", _manywell_std, replace=True)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)


cudnn.benchmark = True


def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):

    try:
        train_utils.setup(cfg)
        print(str(cfg))

        device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        # always set (and so reset any state left by a previous job in this process)
        fast_sampling = cfg.get("fast_sampling", False) and device.startswith("cuda")
        enable_cuda_graph_sampling(fast_sampling)
        if fast_sampling:
            print("Sampling with CUDA graphs.")
        elif cfg.get("fast_sampling", False):
            print("fast_sampling needs CUDA; ignored.")

        fast_train = cfg.get("fast_train", False) and device.startswith("cuda")
        if fast_train:
            print("Training with fused Adam and torch.compile.")
        elif cfg.get("fast_train", False):
            print("fast_train needs CUDA; ignored.")

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Instantiating energy...")
        energy = hydra.utils.instantiate(cfg.energy, device=device)


        print("Instantiating source...")
        source = hydra.utils.instantiate(cfg.source, device=device)


        print('Instantiating model...')
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        controller = hydra.utils.instantiate(cfg.controller).to(device)
        if cfg.get("init_controller"):
            # start from a pre-trained controller (same SDE: same sigma_max); the optimizer starts fresh
            ckpt = torch.load(cfg.init_controller, map_location=device, weights_only=False)
            controller.load_state_dict(ckpt["controller"])
            print(f"Loaded controller from {cfg.init_controller}")
        sde = ControlledSDE(ref_sde, controller).to(device)


        if "corrector" in cfg:
            print('Instantiating corrector & corrector matcher...')
            corrector = hydra.utils.instantiate(cfg.corrector).to(device)
            corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)
        else:
            corrector = corrector_matcher = None


        print("Instantiating grad of costs...")
        grad_term_cost = hydra.utils.instantiate(
            cfg.term_cost,
            corrector=corrector,
            energy=energy,
            ref_sde=ref_sde,
            source=source,
        )


        print("Instantiating adjoint matcher...")
        adjoint_matcher = hydra.utils.instantiate(
            cfg.adjoint_matcher,
            grad_term_cost=grad_term_cost,
            sde=sde,
        )


        print("Instantiating optimizer...")
        lr_schedule = None # TODO(ghliu) add scheduler
        if corrector is not None:
            optimizer = torch.optim.Adam([
                {'params': controller.parameters(), **cfg.adjoint_matcher.optim},
                {'params': corrector.parameters(), **cfg.corrector_matcher.optim},
            ], fused=fast_train)
        else:
            optimizer = torch.optim.Adam(
                controller.parameters(), **cfg.adjoint_matcher.optim, fused=fast_train,
            )

        if cfg.get("lr_warmup_epochs"):
            # linear lr warm-up from lr_warmup_start to the configured lr over lr_warmup_epochs epochs, stepped per iteration; then constant
            assert corrector is None  # checkpoint resume is disabled here (start_epoch = 0), so the schedule starts at step 0
            n_warm = int(cfg.lr_warmup_epochs) * cfg.train_itr_per_epoch
            f0 = float(cfg.lr_warmup_start) / float(cfg.adjoint_matcher.optim.lr)
            lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: f0 + (1.0 - f0) * min(1.0, i / n_warm))

        # compiled wrappers share the parameters; sampling keeps using the eager modules
        train_models = {"adjoint": controller, "corrector": corrector}
        if fast_train:
            train_models = {k: m if m is None else torch.compile(m) for k, m in train_models.items()}

        # EMA of the controller: it samples for evaluation, and for the buffer with ema_buffer
        if cfg.get("ema_decay"):
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            ema = AveragedModel(controller, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
            eval_sde = ControlledSDE(ref_sde, ema.module).to(device)
            if cfg.get("ema_buffer", False):
                adjoint_matcher.sde = eval_sde
        else:
            ema, eval_sde = None, sde


        # Checkpoint loading disabled: every run starts from epoch 0,
        # whatever checkpoint exists in the run folder.
        # checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
        # checkpoint_path.parent.mkdir(exist_ok=True)
        # if checkpoint_path.exists():
        #     print(f"Loading checkpoint from {checkpoint_path}...")
        #     checkpoint = torch.load(checkpoint_path, weights_only=False)  # our own file; contains the cfg
        #     start_epoch = train_utils.load(
        #         checkpoint,
        #         optimizer,
        #         controller,
        #         adjoint_matcher,
        #         corrector=corrector,
        #         corrector_matcher=corrector_matcher,
        #     )
        #     # Note: Not wrapping this in a DDP since we don't differentiate through SDE simulation.
        # else:
        #     start_epoch = 0
        start_epoch = 0


        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )
            if corrector is not None:
                corrector = torch.nn.parallel.DistributedDataParallel(
                    corrector, device_ids=[cfg.gpu], find_unused_parameters=True
                )


        print("Instantiating writer...")
        writer = train_utils.Writer(
            name=cfg.exp_name,
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )


        print("Instantiating evaluator...")
        eval_dir = Path("eval_figs")
        eval_dir.mkdir(exist_ok=True)
        evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)


        if cfg.get("init_buffer"):
            # the initial replay buffer = given endpoints X1 (the SAME set for AS and Barker): the matcher builds the usual entries (adjoint1 = ∇E + ∇log p^base_1
            # of the CURRENT target, orbit, denoised points) from them instead of simulating the SDE
            import adjoint_samplers.components.matcher as matcher_module
            X = torch.load(cfg.init_buffer, map_location="cpu", weights_only=False)["model_samples"][: adjoint_matcher.buffer.buffer_size].to(device)
            if hasattr(adjoint_matcher, "init_wells"):
                print(f"Well bank initialised from the initial buffer: {adjoint_matcher.init_wells(X)} double-well coordinates")
            real_sdeint, chunk = matcher_module.sdeint, 500
            for k in range(0, len(X), chunk):
                xk = X[k:k + chunk]
                matcher_module.sdeint = lambda sde_, x0_, ts_, only_boundary=True, _x=xk: (x0_, _x)
                try:
                    adjoint_matcher.populate_buffer(source.sample([len(xk),]).to(device), train_utils.get_timesteps(**cfg.timesteps).to(device), False)
                finally:
                    matcher_module.sdeint = real_sdeint
            print(f"Initial buffer: {len(adjoint_matcher.buffer)} samples from {cfg.init_buffer}")

        corr_cost, warned_correction, corr_memory = {"E": 0, "grad": 0}, False, None
        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")
        for epoch in range(start_epoch, cfg.num_epochs):
            stage = train_utils.determine_stage(epoch, cfg)

            matcher, model = {
                "adjoint": (adjoint_matcher, train_models["adjoint"]),
                "corrector": (corrector_matcher, train_models["corrector"]),
            }.get(stage)

            if hasattr(matcher, "set_progress"):
                matcher.set_progress(epoch, cfg.num_epochs) # γ annealing of the Barker / MH term

            loss = train_one_epoch(
                matcher,
                model,
                source,
                optimizer,
                lr_schedule,
                epoch,
                device,
                cfg,
                after_step=(lambda: ema.update_parameters(controller)) if ema and stage == "adjoint" else None,
            )

            log_dict = {
                f"{stage}_loss": loss,
                f"{stage}_buffer_size": len(matcher.buffer),
            }
            if hasattr(matcher, "pop_stats"):
                log_dict.update(matcher.pop_stats())
            writer.log(log_dict, step=epoch)

            # Observer (no feedback into training): what the IMH correction would do to the current buffer
            if (cfg.get("correction_every") and stage == "adjoint" and distributed_mode.is_main_process()
                    and epoch >= cfg.correction_warmup
                    and (epoch % cfg.correction_every == 0 or epoch == cfg.num_epochs - 1)):
                dist = getattr(energy, "dist", None)
                if hasattr(dist, "assign") and hasattr(dist, "K"):
                    buf = adjoint_matcher.buffer
                    x1_buf = torch.cat(buf.batches["x1"])[-buf.buffer_size:].to(device)
                    diag, corr_memory = buffer_correction_diagnostic(
                        energy, dist, x1_buf, chains=cfg.correction_chains, steps=cfg.correction_steps,
                        burn=cfg.correction_burn, memory=corr_memory, use_memory=cfg.correction_memory)
                    corr_cost["E"] += diag["n_E_total"]
                    corr_cost["grad"] += diag["n_grad_total"]
                    writer.log({
                        **{f"corr/{k}": v for k, v in diag.items()},
                        "cost/as_grad_evals": (epoch + 1) * cfg.adjoint_matcher.resample_size, # buffer refills, one gradient each
                        "cost/corr_E_cum": corr_cost["E"],
                        "cost/corr_grad_cum": corr_cost["grad"],
                    }, step=epoch)
                    print(f"[correction diagnostic | ep={epoch:04}] buffer mode_tv {diag['buffer_mode_tv']:.3f} -> IMH {diag['imh_mode_tv']:.3f}"
                          f" (acc {diag['accept']:.2f}, ess {diag['ess']:.0f}, centres {diag['M_cover']}, "
                          f"modes with a centre {diag['modes_with_centre_raw']} -> {diag['modes_with_centre_cover']}/{dist.K})"
                          + (f"; with memory: IMH {diag['imh_mode_tv_mem']:.3f}, {diag['modes_with_centre_mem']}/{dist.K} modes" if "imh_mode_tv_mem" in diag else ""))
                elif not warned_correction:
                    print("correction_every needs a GMM target (energy.dist with K and assign); ignored.")
                    warned_correction = True

            print("[{0} | {1}] {2}".format(
                cyan(  f"{stage:<7}"),
                yellow(f"ep={epoch:04}"),
                green( f"loss={loss:.4f}"),
            ))

            # Eval epoch according to the frequency
            # otherwise eval at the end of adjoint matching
            if "eval_freq" in cfg:
                eval_this_epoch = epoch % cfg.eval_freq == 0
            else:
                eval_this_epoch = train_utils.is_last_am_epoch(epoch, cfg)

            if distributed_mode.is_main_process() and eval_this_epoch:
                # eval only after adjoint training
                if stage == "adjoint":
                    n_gen_samples = 0
                    x1_list = []
                    while n_gen_samples < cfg.num_eval_samples:
                        B = min(cfg.eval_batch_size, cfg.num_eval_samples - n_gen_samples)
                        x0 = source.sample([B,]).to(device)
                        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)

                        # model samples
                        x0, x1 = sdeint(
                            eval_sde,
                            x0,
                            timesteps,
                            only_boundary=True,
                        )
                        x1_list.append(x1)
                        n_gen_samples += x1.shape[0]
                        print("Generated {} samples (total: {}/{})".format(
                            x1.shape[0],
                            n_gen_samples,
                            cfg.num_eval_samples,
                        ))

                    samples = torch.cat(x1_list, dim=0)
                    eval_dict = evaluator(samples)
                    if hasattr(evaluator, "mode_metrics"):
                        # mass errors of the replay buffer the regression sees, next to the model's
                        buffer = adjoint_matcher.buffer
                        buffer_x1 = torch.cat(buffer.batches["x1"])[-buffer.buffer_size:].to(device)
                        eval_dict.update({
                            f"buffer/{k}": v
                            for k, v in evaluator.mode_metrics(buffer_x1, per_mode=True).items()
                        })
                    print(f"Evaluated @{epoch=}!")

                    if "hist_img" in eval_dict:
                        eval_dict["hist_img"].save(eval_dir / "gen.png")

                    writer.log(eval_dict, step=epoch)
                    print({k: v for k, v in eval_dict.items() if isinstance(v, float)})
                    if len(log_dict) > 2:
                        print({k: v for k, v in log_dict.items() if "/" in k}) # matcher stats

                # Checkpoint saving disabled: runs never resume from a previous stop.
                # print("Saving checkpoint ... ")
                # train_utils.save(
                #     epoch,
                #     cfg,
                #     optimizer,
                #     controller,
                #     adjoint_matcher,
                #     corrector=corrector,
                #     corrector_matcher=corrector_matcher,
                # )

        if cfg.dump_final and distributed_mode.is_main_process():
            # final replay buffer X1 and fresh samples of the final model, for post-hoc use (scripts/imh_*.py)
            buffer = adjoint_matcher.buffer
            buffer_x1 = torch.cat(buffer.batches["x1"])[-buffer.buffer_size:].detach().cpu()
            x1_list, n_gen = [], 0
            while n_gen < cfg.dump_n_samples:
                B = min(cfg.eval_batch_size, cfg.dump_n_samples - n_gen)
                x0 = source.sample([B,]).to(device)
                timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)
                _, x1 = sdeint(eval_sde, x0, timesteps, only_boundary=True)
                x1_list.append(x1.detach().cpu())
                n_gen += B
            torch.save(
                {"buffer_x1": buffer_x1, "model_samples": torch.cat(x1_list), "epoch": cfg.num_epochs, "exp_name": cfg.exp_name},
                "samples_final.pt",
            )
            print(f"Saved samples_final.pt: buffer {tuple(buffer_x1.shape)}, model samples {cfg.dump_n_samples}")
            # the trained controller (and its EMA), to simulate trajectories after the run (scripts/committed_mass.py)
            torch.save(
                {
                    "controller": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
                    "ema": None if ema is None else {k: v.detach().cpu() for k, v in ema.module.state_dict().items()},
                    "epoch": cfg.num_epochs,
                    "exp_name": cfg.exp_name,
                },
                "controller_final.pt",
            )
            print("Saved controller_final.pt")

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
