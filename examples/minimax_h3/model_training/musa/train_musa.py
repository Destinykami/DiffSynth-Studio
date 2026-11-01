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
    from accelerate.utils import DistributedType

    orig_prepare_model = accelerate.Accelerator.prepare_model

    def prepare_model(self, model, evaluation_mode=False, device_placement=None):
        if self.distributed_type == DistributedType.FSDP and not getattr(self, "is_fsdp2", False):
            fsdp_plugin = self.state.fsdp_plugin
            if getattr(fsdp_plugin, "limit_all_gathers", None) is False:
                fsdp_plugin.limit_all_gathers = True
            saved_mp = self.mixed_precision
            model = orig_prepare_model(self, model, evaluation_mode=evaluation_mode, device_placement=device_placement)
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

    accelerate.Accelerator.prepare_model = prepare_model


def install_lifecycle_and_telemetry() -> None:
    import torch
    import torch.distributed as dist
    import diffsynth.diffusion.logger as ds_logger
    import diffsynth.diffusion.runner as ds_runner

    orig_save_model = ds_logger.ModelLogger.save_model
    orig_on_step_end = ds_logger.ModelLogger.on_step_end
    orig_on_training_end = ds_logger.ModelLogger.on_training_end
    profile_enabled = os.environ.get("TRAIN_STEP_PROFILE", "1") == "1"

    step_timestamps: list[float] = []
    step_losses: list[float] = []
    loop_start_holder = [None]

    if profile_enabled:
        import accelerate

        orig_prepare = accelerate.Accelerator.prepare

        def timed_prepare(self, *args, **kwargs):
            out = orig_prepare(self, *args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            loop_start_holder[0] = time.perf_counter()
            return out

        accelerate.Accelerator.prepare = timed_prepare

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        if profile_enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            step_timestamps.append(now)
            loss_val = kwargs.get("loss")
            if isinstance(loss_val, torch.Tensor):
                step_losses.append(float(loss_val.detach().float().mean().cpu()))
        return orig_on_step_end(self, accelerator, model, save_steps=save_steps, **kwargs)

    def save_model(self, accelerator, model, file_name):
        if self.remove_prefix_in_ckpt is None:
            self.remove_prefix_in_ckpt = "pipe.dit."
        t0 = time.perf_counter()
        orig_save_model(self, accelerator, model, file_name)
        if profile_enabled and accelerator.is_main_process:
            print(f"[train_musa] saved {file_name} in {time.perf_counter() - t0:.2f}s", flush=True)

    def on_training_end(self, accelerator, model, save_steps=None):
        if profile_enabled and step_timestamps and accelerator.is_main_process:
            t0 = loop_start_holder[0] if loop_start_holder[0] is not None else step_timestamps[0]
            dts = [step_timestamps[0] - t0] + [
                step_timestamps[i] - step_timestamps[i - 1] for i in range(1, len(step_timestamps))
            ]
            steady = sum(dts[1:]) / len(dts[1:]) if len(dts) > 1 else dts[0]
            avg_all = sum(dts) / len(dts)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            peak_resv = torch.cuda.max_memory_reserved() / (1024**3)
            dts_str = ", ".join(f"{d:.2f}s" for d in dts)
            loss_str = ", ".join(f"{l:.4f}" for l in step_losses)
            print(
                f"[TRAIN_BENCH] steps={len(dts)} first={dts[0]:.2f}s steady={steady:.2f}s/it "
                f"avg={avg_all:.2f}s/it peak_alloc={peak_alloc:.2f}GiB peak_resv={peak_resv:.2f}GiB "
                f"step_times=[{dts_str}] losses=[{loss_str}]",
                flush=True,
            )
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
