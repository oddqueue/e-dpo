from contextlib import nullcontext
from typing import Callable, Optional, Union, Literal

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.utils import is_torch_xpu_available
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from datasets import Dataset
from trl import DPOTrainer
from config import EpsilonDPOConfig


class EpsilonDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[EpsilonDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
    ):
        assert isinstance(args, EpsilonDPOConfig), "`EpsilonDPOTrainer` requires `EpsilonDPOConfig` for the `args` argument."
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        self.epsilon = args.epsilon
        self.steps = 0.
    
    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output


    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], ref_logits: Optional[torch.FloatTensor]=None):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]
            loss_mask = loss_mask[:, :first_empty_col]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            if self.use_num_logits_to_keep:
                # Compute num_logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                num_logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["num_logits_to_keep"] = num_logits_to_keep

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_num_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -num_logits_to_keep:]
                loss_mask = loss_mask[:, -num_logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Estimating ε-steps
        if ref_logits is not None:
            p_epsilon_logits = ((1 + self.epsilon) * logits) - (self.epsilon * ref_logits)
            p_epsilon_per_token_logps = torch.gather(p_epsilon_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            p_epsilon_per_token_logps[~loss_mask] = 0
            p_epsilon_per_token_logps = torch.roll(p_epsilon_per_token_logps, shifts=1, dims=1)

            n_epsilon_logits = ((1 - self.epsilon) * logits) + (self.epsilon * ref_logits)
            n_epsilon_per_token_logps = torch.gather(n_epsilon_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            n_epsilon_per_token_logps[~loss_mask] = 0
            n_epsilon_per_token_logps = torch.roll(n_epsilon_per_token_logps, shifts=1, dims=1)

            p_epsilon_all_logps = p_epsilon_per_token_logps.sum(-1)
            n_epsilon_all_logps = n_epsilon_per_token_logps.sum(-1)

            logratios = all_logps[:num_examples] - all_logps[num_examples:]
            p_epsilon_logratios = p_epsilon_all_logps[:num_examples] - p_epsilon_all_logps[num_examples:]
            n_epsilon_logratios = n_epsilon_all_logps[:num_examples] - n_epsilon_all_logps[num_examples:]

            p_epsilon_steps = (p_epsilon_logratios > logratios) & (logratios > n_epsilon_logratios)
            n_epsilon_steps = (n_epsilon_logratios > logratios) & (logratios > p_epsilon_logratios)
            steps = 1*p_epsilon_steps - 1*n_epsilon_steps

            output["steps"] = steps
        
        else:
            output["logits"] = logits

        mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output


    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        steps: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the ε-DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.
            steps (`torch.BoolTensor`):
                KL step decision of each example in the batch. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """

        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = logratios - ref_logratios

        updated_beta = self.beta / (1 + self.epsilon * steps)

        losses = (
            -F.logsigmoid(updated_beta * logits) * (1 - self.label_smoothing)
            -F.logsigmoid(updated_beta * logits) * self.label_smoothing
        )

        chosen_rewards = updated_beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = updated_beta * (rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards


    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        ref_model_output = self.compute_ref_log_probs(batch)
        model_output = self.concatenated_forward(model, batch, ref_model_output['logits'])

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_model_output["chosen_logps"],
            ref_model_output["rejected_logps"],
            model_output["steps"],
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        if train_eval == "train":
            self.steps += (model_output["steps"].float().mean() / self.args.gradient_accumulation_steps)

            metrics[f"{prefix}kl/p_epsilon_steps"] = (
            self.accelerator.gather_for_metrics((model_output["steps"] == 1)).float().mean().item()
            )
            metrics[f"{prefix}kl/n_epsilon_steps"] = (
            self.accelerator.gather_for_metrics((model_output["steps"] == -1)).float().mean().item()
            )

            if self.accelerator.gradient_state.sync_gradients:
                mean_steps = self.accelerator.gather(self.steps).mean()

                metrics[f"{prefix}kl/beta"] = (
                    self.beta
                )
                metrics[f"{prefix}kl/avg_steps"] = (
                    mean_steps
                )

                self.beta = (self.beta / (1 + mean_steps * self.epsilon))
                self.steps = 0.

        return losses.mean(), metrics