from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from .noise import add_noise_to_model

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM


def select_layers_from_gradnorm(gradnorm_json, top_k=None, percent=None, mode="high"):
    """Select layer indices from gradnorm probe result JSON."""
    with open(gradnorm_json, "r") as f:
        data = json.load(f)
    layer_grad_norms = data.get("layer_grad_norms", {})
    items = [(int(k), float(v)) for k, v in layer_grad_norms.items() if k.isdigit()]
    items = sorted(items, key=lambda x: x[1], reverse=(mode == "high"))
    
    if len(items) == 0:
        return []
    
    if top_k is not None:
        keep = max(1, min(top_k, len(items)))
        return [lid for lid, _ in items[:keep]]
    
    if percent is not None:
        keep = max(1, int(len(items) * percent / 100.0))
        return [lid for lid, _ in items[:keep]]
    
    return [lid for lid, _ in items]


def freeze_layers_except_selected(model, selected_layer_indices):
    """Freeze all layers except the selected ones."""
    selected_set = set(selected_layer_indices)
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        is_selected = False
        if "layers." in name:
            parts = name.split("layers.")
            if len(parts) > 1:
                layer_id_str = parts[1].split(".")[0]
                if layer_id_str.isdigit():
                    layer_id = int(layer_id_str)
                    if layer_id in selected_set:
                        is_selected = True
        
        if is_selected:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"[layer_finetune] frozen={frozen_count} trainable={trainable_count} layers={sorted(selected_layer_indices)}")


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    alpha: float = 1.0,
    loss_scale: float = 2.0,
    threshold: int = 90,
    add_noise: bool = False,
    outlier_indices_path: str | None = None,
    noise_mode: str = 'row',
    noise_scale: float = 1.0,
    gradnorm_json: str | None = None,
    gradnorm_topk: int | None = None,
    gradnorm_percent: float | None = None,
    gradnorm_mode: str = "high"
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."
    
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_dir=tokenizer_dir)
    
    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type or 'rmu' in loss_type
        else None
    )
    
    dataset = ForgetRetainDataset(data_file, tokenizer=tokenizer, retain_file_path=retain_data_file, max_len=max_len)
    
    selected_layers = None
    if gradnorm_json is not None:
        print(f"[layer_finetune] loading gradnorm from {gradnorm_json}")
        selected_layers = select_layers_from_gradnorm(
            gradnorm_json=gradnorm_json,
            top_k=gradnorm_topk,
            percent=gradnorm_percent,
            mode=gradnorm_mode
        )
        print(f"[layer_finetune] selected {len(selected_layers)} layers: {selected_layers}")
        freeze_layers_except_selected(model, selected_layers)
    else:
        print("[layer_finetune] finetuning all layers")
    
    if add_noise:
        if noise_mode in {'row', 'group'}:
            assert outlier_indices_path is not None and outlier_indices_path.endswith('.npz'), \
                "outlier_indices_path must point to a .npz file"
            group_size = 1 if noise_mode == 'row' else 128
            print(f"[noise] mode={noise_mode} path={outlier_indices_path} scale={noise_scale} group_size={group_size}")
            model = add_noise_to_model(model, npz_path=outlier_indices_path, noise_scale=noise_scale, group_size=group_size)
        else:
            raise ValueError(f"Unknown noise_mode: {noise_mode}. Must be 'row' or 'group'.")
    
    if device_count() == 0:
        raise ValueError("Device not detected!")
    
    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='no',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'
    )
    
    if 'sure' in loss_type:
        raise NotImplementedError("SURE class has been removed. Please use IterativeUnlearner instead.")
    
    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        alpha=alpha,
        loss_scale=loss_scale
    )
    
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)


class IterativeUnlearner(Trainer):
    """Iterative unlearner from the benchmark MUSE."""
    
    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 alpha: float = 0.01,
                 loss_scale: float = 2.0,
                 threshold: int = 90,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta
        self.alpha = alpha
        self.loss_scale = loss_scale
        
        if ref_model is not None:
            ref_model.eval()
        
        super().__init__(*args, **kwargs)
        
        trainable_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        print(f"[trainer_init] trainable_params={len(trainable_params)}")
        
        self._step_count = 0
    
    def compute_loss(self, model, x, return_outputs=False):
        self._step_count += 1
        
        x_f, x_r = x
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss
        
        if 'gdr' in self.loss_type or 'klr' in self.loss_type or 'rmu' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss
        
        if 'klf' in self.loss_type or 'npo' in self.loss_type or 'rmu' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )
        
        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
        
        loss = 0
        
        if 'ga' in self.loss_type:
            loss += -loss_f
        
        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            npo_term = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            loss += npo_term
        
        if 'gdr' in self.loss_type:
            loss += loss_r * self.alpha * self.loss_scale
        
        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")
        
        if 'klr' in self.loss_type:
            if 'avg_klr' in self.loss_type:
                kl_r_per_sample = F.kl_div(
                    F.log_softmax(outputs_r.logits, dim=-1),
                    F.log_softmax(outputs_r_ref.logits, dim=-1),
                    reduction='none',
                    log_target=True
                )
                kl_r = kl_r_per_sample.sum(dim=-1).mean()
            else:
                kl_r = F.kl_div(
                    F.log_softmax(outputs_r.logits, dim=-1),
                    F.log_softmax(outputs_r_ref.logits, dim=-1),
                    reduction='batchmean',
                    log_target=True
                )
            klr_weight = self.alpha * self.loss_scale
            loss += kl_r * klr_weight
        
        if 'rmu' in self.loss_type:
            u = torch.rand(model.config.hidden_size).uniform_(-1, 1).to(outputs_f.logits.device)
            u /= u.norm()
            
            activations_f = model(x_f['input_ids'], output_hidden_states=True).hidden_states[-1]
            activations_r = model(x_r['input_ids'], output_hidden_states=True).hidden_states[-1]
            activations_r_ref = self.ref_model(x_r['input_ids'], output_hidden_states=True).hidden_states[-1]
            
            L_forget = torch.mean(torch.norm(activations_f - self.alpha * u, dim=-1) ** 2)
            L_retain = torch.mean(torch.norm(activations_r - activations_r_ref, dim=-1) ** 2)
            
            loss += L_forget + self.alpha * L_retain
        
        if self._step_count % 10 == 0:
            logs = {'loss_total': float(loss.detach().item())}
            if 'ga' in self.loss_type:
                logs['loss_f'] = float(loss_f.detach().item())
            if 'npo' in self.loss_type:
                logs['loss_npo'] = float(npo_term.detach().item())
            if 'gdr' in self.loss_type:
                logs['loss_r'] = float(loss_r.detach().item())
                logs['loss_r_weighted'] = float((loss_r * self.alpha * self.loss_scale).detach().item())
            if 'klr' in self.loss_type:
                logs['kl_r'] = float(kl_r.detach().item())
                logs['kl_r_weighted'] = float((kl_r * self.alpha * self.loss_scale).detach().item())
            if 'rmu' in self.loss_type:
                logs['L_forget'] = float(L_forget.detach().item())
                logs['L_retain'] = float(L_retain.detach().item())
            self.log(logs)
        
        return (loss, outputs_f) if return_outputs else loss
    
    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
