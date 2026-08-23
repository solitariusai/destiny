from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from taktiny.data.prelude import BatchMap, DatasetUtils, Map
from taktiny.trainer.config import DatasetConfig, TrainingConfig

from taktiny.trainer.loss.preference import dpo_loss, ipo_loss
from taktiny.trainer.trainer import Trainer


@dataclasses.dataclass
class DPODatasetConfig:
    """Dataset configuration for Direct Preference Optimization."""

    tokenizer: Any
    dataset: Any = None
    repo_id: str | None = None
    max_length: int | None = None
    batch_size: int = 4
    drop_remainder: bool = True
    shuffle: bool = True
    seed: int = 42
    epochs: int = 1
    streaming: bool = False
    operations: Sequence[Any] = dataclasses.field(default_factory=list)

    def _tokenize_operation(self) -> Any:
        tokenizer = self.tokenizer
        max_length = self.max_length

        if not callable(getattr(tokenizer, 'apply_chat_template', None)):
            raise ValueError(
                'DPODatasetConfig requires a tokenizer with apply_chat_template'
            )

        def tokenize_batch(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            chosen_convs = []
            rejected_convs = []
            for row in rows:
                if 'chosen' not in row or 'rejected' not in row:
                    raise KeyError('DPO records must have "chosen" and "rejected" fields')
                chosen_convs.append(row['chosen'])
                rejected_convs.append(row['rejected'])

            chosen_full = tokenizer.apply_chat_template(
                chosen_convs,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )
            rejected_full = tokenizer.apply_chat_template(
                rejected_convs,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
            )

            encoded: list[dict[str, Any]] = []
            for c_ids, r_ids in zip(chosen_full, rejected_full, strict=True):
                # Find prompt length (common prefix)
                prompt_length = 0
                for c, r in zip(c_ids, r_ids):
                    if c != r:
                        break
                    prompt_length += 1

                if max_length is not None:
                    c_ids = c_ids[:max_length]
                    r_ids = r_ids[:max_length]
                
                c_ids = np.asarray(c_ids, dtype=np.int32)
                r_ids = np.asarray(r_ids, dtype=np.int32)

                c_labels = c_ids.copy()
                c_labels[:min(prompt_length, len(c_labels))] = -100
                r_labels = r_ids.copy()
                r_labels[:min(prompt_length, len(r_labels))] = -100

                encoded.append({
                    'chosen_input_ids': c_ids,
                    'chosen_labels': c_labels,
                    'chosen_attention_mask': np.ones_like(c_ids, dtype=np.bool_),
                    'rejected_input_ids': r_ids,
                    'rejected_labels': r_labels,
                    'rejected_attention_mask': np.ones_like(r_ids, dtype=np.bool_),
                })
            return encoded

        return BatchMap(tokenize_batch, batch_size=512)

    def _pad_operation(self) -> Any:
        max_length = self.max_length
        tokenizer = self.tokenizer
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = 0

        def pad(record: Mapping[str, Any]) -> dict[str, Any]:
            res = {}
            for prefix in ('chosen', 'rejected'):
                ids = np.asarray(record[f'{prefix}_input_ids'])
                length = len(ids)
                pad_len = max_length - length if max_length is not None and length < max_length else 0
                if max_length is not None and length > max_length:
                    ids = ids[:max_length]
                    length = max_length
                
                padded = np.pad(ids, (0, pad_len), constant_values=pad_id).astype(np.int32)
                mask = np.ones(max_length or length, dtype=np.bool_)
                mask[length:] = False
                labels = np.asarray(record.get(f'{prefix}_labels', ids))
                labels = np.pad(labels[:length], (0, pad_len), constant_values=-100).astype(np.int32)

                res[f'{prefix}_input_ids'] = padded
                res[f'{prefix}_labels'] = labels
                res[f'{prefix}_attention_mask'] = mask
            return res

        return Map(pad)

    def _operations(self) -> list[Any]:
        import grain.python as grain
        ops = list(self.operations)
        ops.append(self._tokenize_operation())
        ops.append(self._pad_operation())
        ops.append(grain.Batch(self.batch_size, drop_remainder=self.drop_remainder))
        return ops

    def build(self) -> Any:
        if self.dataset is not None:
            return DatasetUtils.from_datasets(
                self.dataset,
                operations=self._operations(),
                shuffle=self.shuffle,
                seed=self.seed,
                num_epochs=self.epochs,
            )

        if self.repo_id is not None:
            from datasets import load_dataset
            dataset = load_dataset(self.repo_id, streaming=self.streaming)
            split = dataset['train'] if isinstance(dataset, Mapping) else dataset
            return DatasetUtils.from_datasets(
                split,
                operations=self._operations(),
                shuffle=self.shuffle,
                seed=self.seed,
                num_epochs=self.epochs,
            )
        raise ValueError('DPODatasetConfig requires dataset or repo_id')


def _get_batch_logps(logits: Any, labels: Any) -> Any:
    # shift logits and labels for causal LM
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    
    loss_mask = shifted_labels != -100
    safe_labels = jnp.where(loss_mask, shifted_labels, 0)

    # Upcast to float32 BEFORE log-normalization: DPO is highly sensitive to
    # log-prob precision. Gathering the token logit and subtracting the
    # logsumexp avoids materializing a second full-size (B, L, V) buffer.
    shifted_logits = shifted_logits.astype(jnp.float32)
    token_logits = jnp.take_along_axis(
        shifted_logits, safe_labels[..., None], axis=-1
    ).squeeze(-1)
    token_logps = token_logits - jax.nn.logsumexp(shifted_logits, axis=-1)

    # Mask out padding and prompt tokens
    masked_logps = jnp.where(loss_mask, token_logps, 0.0)

    # Sum over sequence to get sequence logprobs in float32
    return jnp.sum(masked_logps.astype(jnp.float32), axis=-1)


def create_dpo_loss_fn(
    ref_model: Any,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = 'dpo'
) -> Any:
    """Creates a DPO loss function closure capturing the reference model."""
    
    def dpo_step_loss(model: Any, batch: Mapping[str, Any], rng: Any | None = None) -> Any:
        # Concatenate chosen and rejected to run a single forward pass
        input_ids = jnp.concatenate([batch['chosen_input_ids'], batch['rejected_input_ids']], axis=0)
        labels = jnp.concatenate([batch['chosen_labels'], batch['rejected_labels']], axis=0)
        
        # Policy model forward pass (model is the updated model PyTree)
        policy_outputs = model(input_ids, rngs=rng)
        policy_logits = policy_outputs[0] if isinstance(policy_outputs, tuple) else policy_outputs
        if hasattr(policy_logits, 'logits'):
            policy_logits = policy_logits.logits
            
        policy_logps = _get_batch_logps(policy_logits, labels)
        chosen_logps, rejected_logps = jnp.split(policy_logps, 2, axis=0)
        
        # Reference model forward pass (frozen)
        ref_outputs = ref_model(input_ids, rngs=rng)
        ref_logits = ref_outputs[0] if isinstance(ref_outputs, tuple) else ref_outputs
        if hasattr(ref_logits, 'logits'):
            ref_logits = ref_logits.logits
            
        ref_logps = _get_batch_logps(ref_logits, labels)
        ref_chosen_logps, ref_rejected_logps = jnp.split(ref_logps, 2, axis=0)
        
        # Compute loss
        if loss_type == 'ipo':
            loss = ipo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=beta)
        else:
            loss = dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, 
                beta=beta, label_smoothing=label_smoothing
            )
            
        metrics = {
            'margin': jnp.mean((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)),
            'policy_chosen_logps': jnp.mean(chosen_logps),
            'policy_rejected_logps': jnp.mean(rejected_logps),
            'ref_chosen_logps': jnp.mean(ref_chosen_logps),
            'ref_rejected_logps': jnp.mean(ref_rejected_logps),
        }
            
        return loss, metrics

    return dpo_step_loss


from collections import defaultdict
from taktiny.trainer.callbacks import TrainerCallback

from rich import print as rprint

class DPOMetricsCallback(TrainerCallback):
    def on_step_end(self, trainer, logs, **kwargs):
        metrics = trainer._flush_metrics()
        if metrics:
            logs.update(metrics)
            
    def on_log(self, trainer, logs, **kwargs):
        if 'margin' in logs:
            margin = logs['margin']
            policy_chosen = logs['policy_chosen_logps']
            policy_rejected = logs['policy_rejected_logps']
            # We use rich to print aligned metrics without disrupting the progress bar
            rprint(
                f"       [dim]↳[/dim] [yellow]margin:[/yellow] {margin:+.4f} [dim]┃[/dim] "
                f"[blue]chosen:[/blue] {policy_chosen:<7.4f} [dim]┃[/dim] "
                f"[red]rejected:[/red] {policy_rejected:<7.4f}"
            )
            
class DPOTrainer(Trainer):
    """Trainer specialized for Direct Preference Optimization (DPO)."""

    def _record_microbatch_metrics(self, metrics):
        # Called via jax.debug.callback on the host during training
        for k, v in metrics.items():
            self._metrics_accumulator[k].append(float(v))

    def _flush_metrics(self):
        if not self._metrics_accumulator:
            return {}
        averaged = {k: sum(v)/len(v) for k, v in self._metrics_accumulator.items()}
        self._metrics_accumulator.clear()
        return averaged

    def __init__(
        self,
        model: Any,
        training_config: TrainingConfig | None = None,
        dataset_config: DPODatasetConfig | None = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = 'dpo',
        **kwargs: Any,
    ) -> None:
        self._metrics_accumulator = defaultdict(list)
        if dataset_config is None:
            raise ValueError('DPOTrainer requires a DPODatasetConfig')
        if training_config is None:
            training_config = TrainingConfig()

        dataloader = dataset_config.build()
        train_dataset = DatasetConfig(
            dataloader,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
        )
        
        # In JAX, we can simply bind the model's current initialized state to a 
        # callable closure to act as the frozen reference model. If this is a PEFT 
        # model (e.g. LoRA), the parameters are passed exactly as initialized (adapters 
        # might be zero). For a full model, JAX's immutable arrays mean the reference 
        # model natively shares memory with the base weights until the optimizer updates 
        # the policy params, at which point the policy params safely diverge.
        # This completely satisfies "if it's full model copy the whole, if it's LoRA 
        # only copy LoRA" automatically and memory-efficiently.
                
        # In TakTiny/Equinox, the model object itself is a PyTree.
        # We can create a frozen reference model by simply mapping the identity
        # function over the PyTree. JAX array buffers are immutable, so this 
        # shares the exact same memory for the base weights perfectly.
        ref_model = jax.tree_util.tree_map(lambda x: x, model)
            
        loss_fn = create_dpo_loss_fn(
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type
        )

        super().__init__(
            model,
            training_config,
            train_dataset,
            loss_fn=loss_fn,
            loss_has_aux=True,
            **kwargs,
        )

__all__ = ['DPODatasetConfig', 'DPOTrainer']
