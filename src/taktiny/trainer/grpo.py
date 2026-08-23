from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from taktiny.data.prelude import BatchMap, DatasetUtils, Map
from taktiny.trainer.config import DatasetConfig, TrainingConfig
from taktiny.trainer.trainer import Trainer


@dataclasses.dataclass(frozen=True)
class GRPODatasetConfig:
    """Dataset configuration for the inner GRPO optimization step."""
    
    dataset: Any = None
    batch_sharding: Any = None
    batch_size: int = 4
    drop_remainder: bool = True
    shuffle: bool = True
    seed: int = 42
    epochs: int = 1
    streaming: bool = False
    operations: Sequence[Any] = dataclasses.field(default_factory=list)

    @property
    def train_dataloader(self) -> Any:
        import grain.python as grain
        ops = list(self.operations)
        ops.append(grain.Batch(self.batch_size, drop_remainder=self.drop_remainder))
        return DatasetUtils.from_datasets(
            self.dataset,
            operations=ops,
            shuffle=self.shuffle,
            seed=self.seed,
            num_epochs=self.epochs,
        )

    @property
    def validation_dataloader(self) -> Any:
        return None


def _get_batch_logps(
    logits: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
) -> jax.Array:
    """Extracts log probabilities of the given labels (tokens) from the logits."""
    # Upcast logits to float32 BEFORE log_softmax to guarantee precision
    logits = logits.astype(jnp.float32)
    # Padded positions may hold sentinel values (e.g. -100); clamp so the
    # gather stays in bounds, those rows are dropped by loss_mask below.
    safe_labels = jnp.maximum(labels, 0)
    # log p(label) = logit[label] - logsumexp(logits). Gathering directly
    # avoids materializing a second full-size buffer that log_softmax would.
    token_logits = jnp.take_along_axis(
        logits, safe_labels[..., None], axis=-1
    ).squeeze(-1)
    token_logps = token_logits - jax.nn.logsumexp(logits, axis=-1)

    # Return masked token logprobs (B, L)
    return jnp.where(loss_mask, token_logps, 0.0)


def create_grpo_loss_fn(
    beta: float = 0.04,
    clip_ratio: float = 0.2,
) -> Any:
    """Creates a compiled GRPO loss function."""
    
    def grpo_step_loss(params: Any, batch: Any, rng: Any = None) -> Any:
        input_ids = batch['input_ids']
        advantages = batch['advantages']
        old_logps = batch['old_logps']
        ref_logps = batch['ref_logps']
        loss_mask = batch['loss_mask']

        # Labels are input_ids shifted by 1 (autoregressive causal LM)
        shifted_labels = jnp.pad(input_ids[:, 1:], ((0, 0), (0, 1)), constant_values=-100)
        
        # Policy Forward Pass
        # We only compute logprobs for the generated tokens, handled via loss_mask
        logits = params(input_ids).logits
        policy_logps = _get_batch_logps(logits, shifted_labels, loss_mask)
        
        # policy_logps is already float32 from _get_batch_logps. Ensure external inputs are float32.
        old_logps = old_logps.astype(jnp.float32)
        ref_logps = ref_logps.astype(jnp.float32)
        advantages = advantages.astype(jnp.float32)

        # 1. GRPO Clipped Objective (PPO Surrogate)
        # policy_logps and old_logps shape: (B, L)
        # advantages shape: (B,) -> broadcast to (B, L)
        adv_expanded = advantages[:, None]
        
        ratio = jnp.exp(policy_logps - old_logps)
        surr1 = ratio * adv_expanded
        surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_expanded
        grpo_clipped_objective = jnp.minimum(surr1, surr2)
        
        # 2. KL Divergence Penalty
        # Using the standard GRPO KL estimator: exp(ref - policy) - (ref - policy) - 1
        kl = jnp.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1.0
        
        # 3. Total Loss per token
        # We maximize objective, so minimize its negative
        token_loss = -(grpo_clipped_objective - beta * kl)
        
        # Average over valid generated tokens
        valid_tokens = jnp.sum(loss_mask)
        loss = jnp.sum(token_loss * loss_mask) / jnp.maximum(valid_tokens, 1.0)
        
        metrics = {
            'grpo_loss': -jnp.sum(grpo_clipped_objective * loss_mask) / jnp.maximum(valid_tokens, 1.0),
            'kl_div': jnp.sum(kl * loss_mask) / jnp.maximum(valid_tokens, 1.0),
            'clip_fraction': jnp.sum((jnp.abs(ratio - 1.0) > clip_ratio) * loss_mask) / jnp.maximum(valid_tokens, 1.0),
            'ratio': jnp.sum(ratio * loss_mask) / jnp.maximum(valid_tokens, 1.0),
        }
        
        return loss, metrics

    return grpo_step_loss


class GRPOTrainer(Trainer):
    """Trainer specialized for the Inner JIT optimization step of GRPO.
    
    This acts as the optimization engine inside your Rollout Orchestrator.
    """
    
    def __init__(
        self,
        model: Any,
        training_config: TrainingConfig,
        train_dataset: GRPODatasetConfig,
        beta: float = 0.04,
        clip_ratio: float = 0.2,
        **kwargs: Any,
    ) -> None:
        
        loss_fn = create_grpo_loss_fn(
            beta=beta,
            clip_ratio=clip_ratio,
        )
        
        super().__init__(
            model,
            training_config,
            train_dataset,
            loss_fn=loss_fn,
            loss_has_aux=True,
            **kwargs,
        )
