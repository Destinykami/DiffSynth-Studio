"""Training entrypoint shim for MiniMax-H3 on Moore Threads MUSA GPUs.

Responsibilities:
1. torchada: CUDA->MUSA redirection for `torch.cuda.*` (Accelerate spawns plain `python` subprocesses,
   so the shim must live in the training entrypoint).
2. musa_compat: device-placement alignment for the Qwen3-VL vision tower under VRAM offload.
3. torchaudio -> PyAV fallback: decodes MP4 audio tracks when torchaudio's built-in backend cannot.
4. peft torchao dispatcher: bypasses `torchao < 0.16` version error during standard LoRA injection.
5. DataLoader sampler generator: keeps `RandomSampler` RNG on CPU when torchada redirects default generators.
6. advertise nccl: makes `torch.distributed.is_nccl_available()` return True so Accelerate selects
   `nccl` (which torchada translates to MUSA's `mccl`).
7. FSDP bf16 FlatParameter preservation (`FSDP_KEEP_BF16=1`, default): prevents Accelerate from
   upcasting all 33B frozen base weights inside FSDP FlatParameters to fp32 when LoRA is injected,
   cutting FSDP parameter memory in half and eliminating fp32->bf16 cast overhead before `all_gather`.
8. Per-step timing & peak VRAM telemetry (`TRAIN_STEP_PROFILE=1`, default): logs rank-0 first-step,
   steady-state `s/it`, and peak allocated/reserved GiB summary at the end of training.
9. Clean distributed teardown (`destroy_process_group`) after Stage-1 data processing and Stage-2 training.
"""
import os
import runpy
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
TRAIN_PY = HERE.parent / "train.py"

os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
# mate's flash_attention_3 varlen backward crashes with lse shape mismatch on MUSA;
# default training to PyTorch SDPA ("torch") unless the caller explicitly chooses another backend.
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
import torchada  # noqa: F401

sys.path.insert(0, str(HERE))
import musa_compat

musa_compat.apply()


def install_audio_fallback() -> None:
    import numpy as np
    import torch
    import torchaudio

    def load_with_pyav(path):
        import av

        with av.open(path) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise RuntimeError(f"no audio stream in {path}")
            sample_rate = int(streams[0].codec_context.sample_rate)
            frames = [f.to_ndarray() for f in container.decode(audio=0)]
        data = np.asarray(np.concatenate(frames, axis=-1), dtype=np.float32)
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sample_rate

    original = torchaudio.load

    def load(path, *args, **kwargs):
        try:
            return original(path, *args, **kwargs)
        except Exception as exc:
            try:
                waveform, sample_rate = load_with_pyav(path)
                warnings.warn(f"[train_musa] torchaudio failed on {path} ({exc}); decoded with PyAV")
                return waveform, sample_rate
            except Exception:
                raise exc

    torchaudio.load = load


def neutralize_torchao_dispatch() -> None:
    try:
        import peft.tuners.lora.torchao as lora_torchao

        lora_torchao.is_torchao_available = lambda: False
    except Exception as exc:
        warnings.warn(f"[train_musa] could not neutralize torchao dispatch: {exc}")


def fix_sampler_generators() -> None:
    import torch
    from torch.utils.data.sampler import RandomSampler

    original = RandomSampler.__iter__

    def _iter(self):
        generator = getattr(self, "generator", None)
        if generator is not None and getattr(generator, "device", torch.device("cpu")).type != "cpu":
            self.generator = torch.Generator(device="cpu")
        return original(self)

    RandomSampler.__iter__ = _iter


def advertise_nccl() -> None:
    import torch.distributed as dist

    dist.is_nccl_available = lambda: True


def install_compile_hook() -> None:
    if os.environ.get("TRAIN_COMPILE") != "1":
        return
    import accelerate

    original = accelerate.Accelerator.prepare
    mode = os.environ.get("TRAIN_COMPILE_MODE", "default")

    def prepare(self, *args, **kwargs):
        for arg in args:
            pipe = getattr(arg, "pipe", None)
            if pipe is None or getattr(arg, "_musa_compiled", False):
                continue
            t0 = time.time()
            pipe.compile_pipeline(mode=mode, dynamic=True, fullgraph=False, compile_models=["dit"])
            arg._musa_compiled = True
            warnings.warn(f"[train_musa] compiled dit blocks (mode={mode}) in {time.time() - t0:.1f}s")
        return original(self, *args, **kwargs)

    accelerate.Accelerator.prepare = prepare


def install_fsdp_bf16_preservation() -> None:
    if os.environ.get("FSDP_KEEP_BF16", "1") != "1":
        return
    import accelerate
    import torch
    import torch.distributed as dist
    from accelerate.utils import DistributedType

    orig_prepare_model = accelerate.Accelerator.prepare_model
    orig_backward = accelerate.Accelerator.backward
    separate_lora = os.environ.get("FSDP_SEPARATE_LORA", "1") == "1"

    def prepare_model(self, model, evaluation_mode=False, device_placement=None):
        if self.distributed_type == DistributedType.FSDP and not getattr(self, "is_fsdp2", False):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            fsdp_plugin = self.state.fsdp_plugin
            if getattr(fsdp_plugin, "limit_all_gathers", None) is False:
                fsdp_plugin.limit_all_gathers = True
            saved_mp = self.mixed_precision
            if separate_lora:
                # Exclude all LoRA sub-modules (lora_A / lora_B ModuleDicts) from FSDP FlatParameters:
                # 1. Frozen 33B base FlatParameters have requires_grad=False, so FSDP skips allocating
                #    1.3 GB unsharded grad buffers per block and skips 50 x 1.3 GB (65 GB/step) reduce_scatters!
                # 2. Replicated LoRA parameters (~310 MB) are kept in fp32 for full master-weight precision
                #    and synchronized across ranks before training and via a single bucketed all_reduce after backward.
                lora_modules = [
                    mod
                    for name, mod in model.named_modules()
                    if name.endswith(".lora_A") or name.endswith(".lora_B")
                ]
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if lora_modules and trainable_params:
                    with torch.no_grad():
                        for p in trainable_params:
                            if p.device.type == "cpu":
                                p.data = p.data.to(self.device)
                            if p.dtype != torch.float32:
                                p.data = p.data.to(torch.float32)
                            if dist.is_initialized() and dist.get_world_size() > 1:
                                dist.broadcast(p.data, src=0)
                    fsdp_plugin.ignored_modules = lora_modules
                    self._fsdp_replicated_lora_params = trainable_params
                    if os.environ.get("DIFFSYNTH_FUSED_LORA_LINEAR", "1") == "1":
                        from diffsynth.diffusion.training_module import _install_fused_peft_lora_linear

                        _install_fused_peft_lora_linear(model)
                    if self.is_main_process:
                        total_mb = sum(p.numel() * p.element_size() for p in trainable_params) / (1024**2)
                        print(
                            f"[train_musa] Excluded {len(lora_modules)} LoRA modules ({len(trainable_params)} fp32 tensors, "
                            f"{total_mb:.1f} MiB) from FSDP FlatParameters (eliminated 65 GB/step frozen grad reduce_scatter).",
                            flush=True,
                        )
            model = orig_prepare_model(self, model, evaluation_mode=evaluation_mode, device_placement=device_placement)

            reverted = 0
            for fsdp_mod in FSDP.fsdp_modules(model):
                if not getattr(fsdp_mod, "_has_params", False):
                    continue
                param = fsdp_mod._flat_param
                if saved_mp == "bf16" and param.dtype == torch.float32:
                    param.data = param.data.to(torch.bfloat16)
                    fsdp_mod._handle._orig_param_dtype = torch.bfloat16
                    reverted += 1
            if reverted and self.is_main_process:
                print(
                    f"[train_musa] Kept {reverted} FSDP FlatParameters in bfloat16 (avoided 2x fp32 base-weight bloat).",
                    flush=True,
                )
            return model
        return orig_prepare_model(self, model, evaluation_mode=evaluation_mode, device_placement=device_placement)

    def backward(self, loss, **kwargs):
        orig_backward(self, loss, **kwargs)
        replicated_lora = getattr(self, "_fsdp_replicated_lora_params", None)
        if replicated_lora and self.sync_gradients and dist.is_initialized() and dist.get_world_size() > 1:
            grads = [p.grad for p in replicated_lora if p.grad is not None]
            if grads:
                flat_grad = torch.cat([g.reshape(-1) for g in grads])
                dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG)
                self._last_lora_grad_norm = float(torch.linalg.vector_norm(flat_grad).detach().cpu())
                offset = 0
                for g in grads:
                    numel = g.numel()
                    g.copy_(flat_grad[offset : offset + numel].view_as(g))
                    offset += numel

    accelerate.Accelerator.prepare_model = prepare_model
    accelerate.Accelerator.backward = backward


def install_lifecycle_and_telemetry() -> None:
    import csv
    import torch
    import torch.distributed as dist
    import diffsynth.diffusion.logger as ds_logger
    import diffsynth.diffusion.loss as ds_loss
    import diffsynth.diffusion.runner as ds_runner

    orig_save_model = ds_logger.ModelLogger.save_model
    orig_on_step_end = ds_logger.ModelLogger.on_step_end
    orig_on_training_end = ds_logger.ModelLogger.on_training_end
    orig_h3_loss = ds_loss.FlowMatchSFTMiniMaxH3AudioVideoLoss
    profile_enabled = os.environ.get("TRAIN_STEP_PROFILE", "1") == "1"
    eval_every = int(os.environ.get("TRAIN_EVAL_EVERY", "0"))
    eval_timesteps = [int(x) for x in os.environ.get("TRAIN_EVAL_TIMESTEPS", "200,500,800").split(",") if x.strip()]

    step_timestamps: list[float] = []
    step_losses: list[float] = []
    step_ema_losses: list[float] = []
    step_grad_norms: list[float] = []
    step_alloc_gbs: list[float] = []
    step_resv_gbs: list[float] = []
    eval_history: list[tuple[int, float, dict[int, float]]] = []
    anchor_holder: list[dict | None] = [None]
    forced_eval_state: dict = {"active": False, "t_idx": None, "noise_v": None, "noise_a": None}
    loop_start_holder = [None]

    if eval_every > 0:
        import copy

        def _clone_tree(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().clone()
            if isinstance(obj, dict):
                return {k: _clone_tree(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return tuple(_clone_tree(v) for v in obj)
            if isinstance(obj, list):
                return [_clone_tree(v) for v in obj]
            return copy.deepcopy(obj)

        def wrapped_h3_loss(pipe, training_cfg_scale: float = 1.0, audio_loss_weight: float = 1.0, inputs_nega=None, **inputs):
            if forced_eval_state["active"]:
                t_idx = forced_eval_state["t_idx"]
                tid = torch.tensor([t_idx], dtype=torch.long)
                t_vid = pipe.scheduler.timesteps[tid].to(dtype=torch.float32, device=pipe.device)
                t_aud = pipe.scheduler_audio.timesteps[tid].to(dtype=torch.float32, device=pipe.device)
                in_v = inputs["input_latents"]
                noise_v = forced_eval_state["noise_v"].to(device=in_v.device, dtype=in_v.dtype)
                inputs["video_latents"] = pipe.scheduler.add_noise(in_v, noise_v, t_vid)
                target_v = pipe.scheduler.training_target(in_v, noise_v, t_vid)
                has_audio = "audio_input_latents" in inputs and inputs["audio_input_latents"] is not None
                if has_audio:
                    in_a = inputs["audio_input_latents"]
                    noise_a = forced_eval_state["noise_a"].to(device=in_a.device, dtype=in_a.dtype)
                    inputs["audio_latents"] = pipe.scheduler_audio.add_noise(in_a, noise_a, t_aud)
                    target_a = pipe.scheduler_audio.training_target(in_a, noise_a, t_aud)
                inputs["use_gradient_checkpointing"] = False
                inputs["use_gradient_checkpointing_offload"] = False
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                pred_v, pred_a = pipe.model_fn(**models, **inputs, timestep_video=t_vid, timestep_audio=t_aud)
                loss = torch.nn.functional.mse_loss(pred_v.float(), target_v.float()) * pipe.scheduler.training_weight(t_vid)
                if has_audio:
                    loss_a = (
                        torch.nn.functional.mse_loss(pred_a.float(), target_a.float())
                        * pipe.scheduler_audio.training_weight(t_aud)
                        * audio_loss_weight
                    )
                    loss = loss + loss_a
                return loss
            return orig_h3_loss(
                pipe,
                training_cfg_scale=training_cfg_scale,
                audio_loss_weight=audio_loss_weight,
                inputs_nega=inputs_nega,
                **inputs,
            )

        ds_loss.FlowMatchSFTMiniMaxH3AudioVideoLoss = wrapped_h3_loss
        import diffsynth.diffusion as ds_pkg

        ds_pkg.FlowMatchSFTMiniMaxH3AudioVideoLoss = wrapped_h3_loss
        try:
            import sys as _sys

            for _m in list(_sys.modules.values()):
                if getattr(_m, "FlowMatchSFTMiniMaxH3AudioVideoLoss", None) is orig_h3_loss:
                    _m.FlowMatchSFTMiniMaxH3AudioVideoLoss = wrapped_h3_loss
        except Exception:
            pass

    def run_fixed_anchor_eval(accelerator, model, step_num: int):
        anchor = anchor_holder[0]
        if anchor is None:
            return
        per_t: dict[int, float] = {}
        forced_eval_state["active"] = True
        forced_eval_state["noise_v"] = anchor["noise_v"]
        forced_eval_state["noise_a"] = anchor["noise_a"]
        try:
            with torch.no_grad():
                for t_idx in eval_timesteps:
                    forced_eval_state["t_idx"] = t_idx
                    sample_inputs = _clone_tree(anchor["raw_inputs"])
                    loss_t = model({}, inputs=sample_inputs).detach().float().mean()
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                    per_t[t_idx] = float(loss_t.cpu())
        finally:
            forced_eval_state["active"] = False
        mean_eval = sum(per_t.values()) / len(per_t)
        eval_history.append((step_num, mean_eval, per_t))
        if accelerator.is_main_process:
            detail = " ".join(f"t{k}={v:.4f}" for k, v in per_t.items())
            print(f"[FIXED_EVAL step={step_num:03d}] mean_loss={mean_eval:.4f} ({detail})", flush=True)

    if profile_enabled:
        import accelerate

        orig_prepare = accelerate.Accelerator.prepare

        def timed_prepare(self, *args, **kwargs):
            out = orig_prepare(self, *args, **kwargs)
            if eval_every > 0:
                for item in (out if isinstance(out, tuple) else (out,)):
                    if isinstance(item, torch.utils.data.DataLoader) and anchor_holder[0] is None:
                        first_batch = next(iter(item))
                        in_shared = first_batch[0] if isinstance(first_batch, (tuple, list)) else {}
                        in_v = in_shared.get("input_latents")
                        if isinstance(in_v, torch.Tensor):
                            gen_v = torch.Generator(device="cpu").manual_seed(12345)
                            noise_v = torch.randn(in_v.shape, generator=gen_v, dtype=torch.float32)
                            in_a = in_shared.get("audio_input_latents")
                            noise_a = None
                            if isinstance(in_a, torch.Tensor):
                                gen_a = torch.Generator(device="cpu").manual_seed(67890)
                                noise_a = torch.randn(in_a.shape, generator=gen_a, dtype=torch.float32)
                            anchor_holder[0] = {
                                "raw_inputs": _clone_tree(first_batch),
                                "noise_v": noise_v,
                                "noise_a": noise_a,
                            }
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            loop_start_holder[0] = time.perf_counter()
            return out

        accelerate.Accelerator.prepare = timed_prepare

    prev_step_end_holder = [None]

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        cur_step = None
        if profile_enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            t_start = prev_step_end_holder[0] if prev_step_end_holder[0] is not None else (loop_start_holder[0] or now)
            step_timestamps.append(now - t_start)
            loss_val = kwargs.get("loss")
            if isinstance(loss_val, torch.Tensor):
                l_t = loss_val.detach().float().mean()
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(l_t, op=dist.ReduceOp.AVG)
                cur_loss = float(l_t.cpu())
                step_losses.append(cur_loss)
                prev_ema = step_ema_losses[-1] if step_ema_losses else cur_loss
                step_ema_losses.append(0.9 * prev_ema + 0.1 * cur_loss)
            step_grad_norms.append(float(getattr(accelerator, "_last_lora_grad_norm", 0.0)))
            step_alloc_gbs.append(torch.cuda.memory_allocated() / (1024**3))
            step_resv_gbs.append(torch.cuda.memory_reserved() / (1024**3))
            cur_step = len(step_timestamps)
            if eval_every > 0 and (cur_step == 1 or cur_step % eval_every == 0):
                run_fixed_anchor_eval(accelerator, model, cur_step)
        ret = orig_on_step_end(self, accelerator, model, save_steps=save_steps, **kwargs)
        if profile_enabled and cur_step is not None and accelerator.is_main_process:
            extra_metrics = {
                "train/step_time_s": step_timestamps[-1],
                "train/alloc_gib": step_alloc_gbs[-1],
                "train/resv_gib": step_resv_gbs[-1],
            }
            if step_ema_losses:
                extra_metrics["train/ema_loss"] = step_ema_losses[-1]
            if step_grad_norms and step_grad_norms[-1] > 0:
                extra_metrics["train/grad_norm"] = step_grad_norms[-1]
            if eval_history and eval_history[-1][0] == cur_step:
                _, mean_eval, per_t = eval_history[-1]
                extra_metrics["eval/fixed_anchor_mean_loss"] = mean_eval
                for t_k, t_v in per_t.items():
                    extra_metrics[f"eval/fixed_anchor_t{t_k}"] = t_v
            for lg in getattr(self, "loggers", []):
                if type(lg).__name__ == "SwanLabLogger":
                    lg.log(extra_metrics, None, self.num_steps)
        if profile_enabled:
            torch.cuda.synchronize()
            prev_step_end_holder[0] = time.perf_counter()
        return ret

    def save_model(self, accelerator, model, file_name):
        if self.remove_prefix_in_ckpt is None:
            self.remove_prefix_in_ckpt = "pipe.dit."
        t0 = time.perf_counter()
        orig_save_model(self, accelerator, model, file_name)
        if profile_enabled and accelerator.is_main_process:
            print(f"[train_musa] saved {file_name} in {time.perf_counter() - t0:.2f}s", flush=True)

    def on_training_end(self, accelerator, model, save_steps=None):
        if profile_enabled and step_timestamps and accelerator.is_main_process:
            dts = step_timestamps
            steady = sum(dts[1:]) / len(dts[1:]) if len(dts) > 1 else dts[0]
            avg_all = sum(dts) / len(dts)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            peak_resv = torch.cuda.max_memory_reserved() / (1024**3)
            dts_str = ", ".join(f"{d:.2f}s" for d in dts[:12]) + (" ..." if len(dts) > 12 else "")
            loss_str = ", ".join(f"{l:.4f}" for l in step_losses[:12]) + (" ..." if len(step_losses) > 12 else "")
            print(
                f"[TRAIN_BENCH] steps={len(dts)} first={dts[0]:.2f}s steady={steady:.2f}s/it "
                f"avg={avg_all:.2f}s/it peak_alloc={peak_alloc:.2f}GiB peak_resv={peak_resv:.2f}GiB "
                f"step_times=[{dts_str}] losses=[{loss_str}]",
                flush=True,
            )
            if self.output_path:
                os.makedirs(self.output_path, exist_ok=True)
                eval_map = {s: (m, pt) for s, m, pt in eval_history}
                metrics_path = os.path.join(self.output_path, "stability_metrics.csv")
                fieldnames = [
                    "step",
                    "step_time_s",
                    "raw_loss",
                    "ema_loss",
                    "grad_norm",
                    "alloc_gib",
                    "resv_gib",
                    "fixed_eval_mean",
                ] + [f"fixed_eval_t{t}" for t in eval_timesteps]
                with open(metrics_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for idx, dt_s in enumerate(dts, start=1):
                        row = {
                            "step": idx,
                            "step_time_s": f"{dt_s:.4f}",
                            "raw_loss": f"{step_losses[idx - 1]:.6f}" if idx <= len(step_losses) else "",
                            "ema_loss": f"{step_ema_losses[idx - 1]:.6f}" if idx <= len(step_ema_losses) else "",
                            "grad_norm": f"{step_grad_norms[idx - 1]:.6f}" if idx <= len(step_grad_norms) else "",
                            "alloc_gib": f"{step_alloc_gbs[idx - 1]:.3f}" if idx <= len(step_alloc_gbs) else "",
                            "resv_gib": f"{step_resv_gbs[idx - 1]:.3f}" if idx <= len(step_resv_gbs) else "",
                            "fixed_eval_mean": "",
                        }
                        if idx in eval_map:
                            m_val, pt = eval_map[idx]
                            row["fixed_eval_mean"] = f"{m_val:.6f}"
                            for t in eval_timesteps:
                                row[f"fixed_eval_t{t}"] = f"{pt.get(t, 0.0):.6f}"
                        writer.writerow(row)
                print(f"[train_musa] wrote per-step stability telemetry to {metrics_path}", flush=True)
        orig_on_training_end(self, accelerator, model, save_steps=save_steps)
        if dist.is_initialized():
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            dist.destroy_process_group()

    ds_logger.ModelLogger.on_step_end = on_step_end
    ds_runner.ModelLogger.on_step_end = on_step_end
    ds_logger.ModelLogger.save_model = save_model
    ds_runner.ModelLogger.save_model = save_model
    ds_logger.ModelLogger.on_training_end = on_training_end
    ds_runner.ModelLogger.on_training_end = on_training_end

    orig_launch_data_process = ds_runner.launch_data_process_task

    def launch_data_process_task(accelerator, dataset, model, model_logger, *args, **kwargs):
        try:
            return orig_launch_data_process(accelerator, dataset, model, model_logger, *args, **kwargs)
        finally:
            if dist.is_initialized():
                accelerator.wait_for_everyone()
                torch.cuda.synchronize()
                dist.destroy_process_group()

    ds_runner.launch_data_process_task = launch_data_process_task
    import diffsynth.diffusion as ds_pkg

    ds_pkg.launch_data_process_task = launch_data_process_task


install_audio_fallback()
neutralize_torchao_dispatch()
fix_sampler_generators()
advertise_nccl()
install_compile_hook()
install_fsdp_bf16_preservation()
install_lifecycle_and_telemetry()

runpy.run_path(str(TRAIN_PY), run_name="__main__")
