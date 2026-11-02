import os, torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedType


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        self.enabled = False
        self.swanlab = None
        try:
            import swanlab

            self.swanlab = swanlab
        except Exception as exc:
            print(f"[SwanLabLogger] swanlab package is not available ({exc}); skipping SwanLab logging.")
            return

        project_name = os.environ.pop("SWANLAB_PROJECT", None) or os.environ.get("SWANLAB_PROJECT_NAME", project_name)
        run_name = os.environ.pop("SWANLAB_RUN_NAME", None) or os.environ.get("SWANLAB_EXPERIMENT_NAME", None)
        mode = os.environ.get("SWANLAB_MODE", None)
        api_key = os.environ.get("SWANLAB_API_KEY", None)

        try:
            if api_key:
                try:
                    self.swanlab.login(api_key=api_key, save=False)
                except Exception as login_exc:
                    print(f"[SwanLabLogger] API key login failed ({login_exc}); falling back to offline/local mode.")
                    if mode is None:
                        mode = "offline"
            init_kwargs = {"project": project_name, "logdir": log_dir}
            if run_name:
                init_kwargs["name"] = run_name
            if mode:
                init_kwargs["mode"] = mode
            try:
                self.swanlab.init(**init_kwargs)
            except Exception as init_exc:
                if mode not in ("offline", "local", "disabled"):
                    print(f"[SwanLabLogger] Online init failed ({init_exc}); falling back to mode='offline'.")
                    init_kwargs["mode"] = "offline"
                    self.swanlab.init(**init_kwargs)
                else:
                    raise
            self.enabled = True
            print(f"SwanLab is enabled. Project: {project_name} (mode={init_kwargs.get('mode', 'online')})")
        except Exception as exc:
            print(f"[SwanLabLogger] Could not initialize SwanLab ({exc}); continuing training without SwanLab.")
            self.enabled = False

    def log(self, key, value, step):
        if not self.enabled or self.swanlab is None:
            return
        try:
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            if isinstance(key, dict):
                self.swanlab.log(key, step=step)
            else:
                self.swanlab.log({key: value}, step=step)
        except Exception as exc:
            print(f"[SwanLabLogger] Warning: failed to log step {step} ({exc}); disabling SwanLab for remaining steps.")
            self.enabled = False

    def close(self):
        if not self.enabled or self.swanlab is None:
            return
        try:
            self.swanlab.finish()
        except Exception as exc:
            print(f"[SwanLabLogger] Warning: error during swanlab.finish() ({exc}).")
        finally:
            self.enabled = False


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import wandb
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, dir=log_dir)
        print(f"Wandb is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.wandb.log({key: value}, step=step)

    def close(self):
        self.wandb.finish()

class CSVLogger:
    def __init__(self, log_dir):
        import csv
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "loss.csv")
        file_exists = os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        self.file = open(self.path, "a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=["step", "key", "value"], lineterminator="\n")
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, key, value, step):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        self.writer.writerow({"step": step, "key": key, "value": value})
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=False,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
        enable_csv_log=False,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        self.enable_csv_log = enable_csv_log
        self.loggers = []
        self.loggers_initialized = False

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            self.loggers.append(WandbLogger(project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log")))
        if self.enable_csv_log:
            self.loggers.append(CSVLogger(self.output_path))
        self.loggers_initialized = True

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        need_init = not self.loggers_initialized
        if accelerator.is_main_process:
            if need_init:
                self.init_loggers()
            loss = kwargs.get("loss")
            if loss is not None:
                for logger in self.loggers:
                    logger.log("loss", loss, self.num_steps)
        else:
            self.loggers_initialized = True
        if need_init and (self.enable_swanlab_log or self.enable_wandb_log):
            accelerator.wait_for_everyone()
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        accelerator.wait_for_everyone()
        if dist.is_available() and dist.is_initialized():
            torch.cuda.synchronize()
            dist.destroy_process_group()
        for logger in self.loggers:
            logger.close()

    @staticmethod
    def _gather_fsdp_trainable_state_dict(accelerator: Accelerator, model: torch.nn.Module, save_dtype=torch.bfloat16):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        state_dict = {}
        module_to_clean_prefix = {}
        for mod_name, mod in model.named_modules():
            if isinstance(mod, FSDP):
                clean = mod_name.replace("_fsdp_wrapped_module.", "").replace("_fsdp_wrapped_module", "").strip(".")
                module_to_clean_prefix[mod] = (clean + ".") if clean else ""

        for fsdp_mod in FSDP.fsdp_modules(model):
            if not getattr(fsdp_mod, "_has_params", False):
                continue
            handle = fsdp_mod._handle
            flat_param = handle.flat_param
            if not flat_param.requires_grad or not hasattr(flat_param, "_params"):
                continue
            pg = handle.process_group
            world_size = handle.world_size
            prefix = module_to_clean_prefix.get(fsdp_mod, "")
            for param, fqn, shape in zip(flat_param._params, flat_param._fqns, flat_param._shapes):
                if not param.requires_grad:
                    continue
                target_dtype = save_dtype if param.is_floating_point() and save_dtype is not None else param.dtype
                local_flat = param.detach().to(dtype=target_dtype).contiguous().view(-1)
                local_n = torch.tensor([local_flat.numel()], dtype=torch.long, device=local_flat.device)
                sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
                dist.all_gather(sizes, local_n, group=pg)
                counts = [int(s.item()) for s in sizes]
                max_n = max(counts)
                if max_n == 0:
                    continue
                padded = torch.zeros(max_n, dtype=target_dtype, device=local_flat.device)
                if local_flat.numel() > 0:
                    padded[: local_flat.numel()].copy_(local_flat)
                gathered = [torch.empty(max_n, dtype=target_dtype, device=local_flat.device) for _ in range(world_size)]
                dist.all_gather(gathered, padded, group=pg)
                if accelerator.is_main_process:
                    pieces = [g[:c] for g, c in zip(gathered, counts) if c > 0]
                    full_tensor = torch.cat(pieces, dim=0)[: shape.numel()].view(shape).cpu()
                    state_dict[f"{prefix}{fqn}"] = full_tensor
                del padded, gathered

        # Also include any replicated trainable parameters excluded from FSDP sharding via ignored_states.
        sharded_param_ids = {
            id(p)
            for fsdp_mod in FSDP.fsdp_modules(model)
            if getattr(fsdp_mod, "_has_params", False) and hasattr(fsdp_mod._handle.flat_param, "_params")
            for p in fsdp_mod._handle.flat_param._params
        }
        if accelerator.is_main_process:
            for name, param in model.named_parameters():
                if not param.requires_grad or id(param) in sharded_param_ids:
                    continue
                clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_fsdp_wrapped_module", "").strip(".")
                target_dtype = save_dtype if param.is_floating_point() and save_dtype is not None else param.dtype
                state_dict[clean_name] = param.detach().to(dtype=target_dtype).cpu()
        return state_dict

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.distributed_type == DistributedType.FSDP and not getattr(accelerator, "is_fsdp2", False):
            save_dtype = getattr(getattr(unwrapped, "pipe", None), "torch_dtype", torch.bfloat16)
            state_dict = self._gather_fsdp_trainable_state_dict(accelerator, model, save_dtype=save_dtype)
            if accelerator.is_main_process:
                if self.remove_prefix_in_ckpt is not None:
                    state_dict = {
                        (k[len(self.remove_prefix_in_ckpt):] if k.startswith(self.remove_prefix_in_ckpt) else k): v
                        for k, v in state_dict.items()
                    }
                state_dict = self.state_dict_converter(state_dict)
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                accelerator.save(state_dict, path, safe_serialization=True)
            accelerator.wait_for_everyone()
            return
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = unwrapped.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
        accelerator.wait_for_everyone()
