# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>
# Added support for GLM 4.6 and other models
# Customized by AlexH from llmresearch.net
# Soupport: https://llmresearch.net/threads/heretic-llm-universal-support-for-new-models-via-dynamic-auto-registration.275/

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = None
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        # --- Patch adaptiv pentru GLM-4.6V ---
        self._patch_glm4v_support()
        # --- Sfârșitul patch-ului ---

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    torch_dtype=dtype,
                    device_map=settings.device_map,
                    trust_remote_code=self.trusted_models.get(settings.model, settings.trust_remote_code),
                )

                if self.trusted_models.get(settings.model) is None:
                    self.trusted_models[settings.model] = True

                # Skip test generation for multimodal models like GLM-4.6V
                if "GLM-4.6V" not in settings.model:
                    self.generate(["Test"], max_new_tokens=1)
                
                print("[green]Ok[/]")
                break
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

    def _patch_glm4v_support(self):
        """Inspectează configurarea modelului și înregistrează dinamic clasele GLM-4V."""
        if "GLM-4.6V" not in self.settings.model:
            return

        try:
            config = AutoConfig.from_pretrained(
                self.settings.model,
                trust_remote_code=self.trusted_models.get(self.settings.model, self.settings.trust_remote_code),
            )
            
            # Extrage numele claselor din configurație
            config_class_name = config.__class__.__name__
            architectures = getattr(config, 'architectures', [])
            
            if architectures:
                model_class_name = architectures[0]
                print(f"(Found GLM-4.6V config: {config_class_name}, model: {model_class_name})... ", end="")
                
                # Încearcă să importeze dinamic clasele necesare
                try:
                    # Importă clasa de configurație
                    config_module = __import__(
                        f"transformers.models.glm4v.configuration_glm4v",
                        fromlist=['configuration_glm4v']
                    )
                    config_class = getattr(config_module, config_class_name)
                    
                    # Importă clasa de modelare
                    modeling_module = __import__(
                        f"transformers.models.glm4v.modeling_glm4v",
                        fromlist=['modeling_glm4v']
                    )
                    model_class = getattr(modeling_module, model_class_name)
                    
                    # Înregistrează clasa de model
                    AutoModelForCausalLM.register(config_class, model_class)
                    print(f"Registered successfully... ", end="")
                    
                except (ImportError, AttributeError) as e:
                    print(f"[red]Failed to register ({e}). Falling back to default behavior.[/]")
                    # Nu putem face mai mult, dar lăsăm `heretic` să încerce
                    # cu `AutoModelForCausalLM` standard, care poate eșua, dar vom oferi
                    # un mesaj de eroare clar
                    
        except Exception as e:
            print(f"[red]Failed to inspect model config ({e}).[/]")

    def reload_model(self):
        dtype = self.model.dtype
        self.model = None
        empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            torch_dtype=dtype,
            device_map=self.settings.device_map,
            trust_remote_code=self.trusted_models.get(self.settings.model, self.settings.trust_remote_code),
        )

        if self.trusted_models.get(self.settings.model) is None:
            self.trusted_models[self.settings.model] = True

    def get_layers(self) -> ModuleList:
        # Calea pentru modelele GLM-4.6V incarcate corect
        if "GLM-4.6V" in self.settings.model:
            with suppress(AttributeError):
                return self.model.model.layers
        
        # Calea pentru majoritatea modelelor multimodale.
        with suppress(AttributeError):
            return self.model.model.language_model.layers

        # Calea pentru modelele text-only.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            if hasattr(matrix, "data") and torch.is_tensor(matrix.data):
                matrix = matrix.data
            assert torch.is_tensor(matrix)
            if component not in matrices:
                matrices[component] = []
            matrices[component].append(matrix)

        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight)

        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear.weight)

        assert matrices.get("mlp.down_proj"), "MLP down-projection not found in layer."
        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]
                distance = abs(layer_index - params.max_weight_position)

                if distance > params.min_weight_distance:
                    continue

                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    device_projector = projector.to(matrix.device)
                    matrix.sub_(weight * (device_projector @ matrix))

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]
        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        if self.response_prefix:
            chat_prompts = [prompt + self.response_prefix for prompt in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

    def get_responses(self, prompts: list[str]) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )
        return self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :])

    def get_responses_batched(self, prompts):
        responses = []
        for batch in batchify(prompts, self.settings.batch_size):
            batch = list(batch)
            for response in self.get_responses(batch):
                responses.append(response)
        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        hidden_states = outputs.hidden_states[0]
        residuals = torch.stack(
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []
        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))
        return torch.cat(residuals, dim=0)

    def get_logprobs(self, prompts: list[str]) -> Tensor:
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logits = outputs.scores[0]
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        logprobs = []
        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))
        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )
        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
