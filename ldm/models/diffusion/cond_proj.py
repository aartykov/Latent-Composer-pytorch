# Copyright 2023 Kakao Brain and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

#from diffusers.src.diffusers.configuration_utils import ConfigMixin, register_to_config
#from ...models import ModelMixin


def project(input_size, intermediate_size, out_size):
    #B, H, W = input.shape
    
    out = nn.Sequential(
        nn.Linear(input_size, intermediate_size), # apply a fully connected layer with output size intermediate_size
        nn.SiLU(),
        nn.Linear(intermediate_size, out_size) # apply a fully connected layer with output size out_size
    )
    return out



class UnCLIPTextProjModel(nn.Module):
    """
    Utility class for CLIP embeddings. Used to combine the image and text embeddings into a format usable by the
    decoder.
    For more details, see the original paper: https://arxiv.org/abs/2204.06125 section 2.1
    """

    #@register_to_config
    def __init__(
        self,
        *,
        clip_extra_context_tokens: int = 4,
        clip_embeddings_dim: int = 768,
        color_embeddings_dim: int = 156,
        time_embed_dim: int,
        cross_attention_dim: int
    ):
        super().__init__()

        self.learned_classifier_free_guidance_embeddings = nn.Parameter(torch.zeros(clip_embeddings_dim))

        # parameters for additional clip time embeddings
        self.embedding_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
        self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(clip_embeddings_dim, time_embed_dim)
        self.color_embeddings_project_to_time_embeddings = nn.Linear(color_embeddings_dim, time_embed_dim)

        # parameters for encoder hidden states
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.clip_extra_context_tokens_proj = project(clip_embeddings_dim, 1024, self.clip_extra_context_tokens * cross_attention_dim) #nn.Linear(
        self.color_extra_context_tokens_proj = project(color_embeddings_dim, 1024, self.clip_extra_context_tokens * cross_attention_dim)
        #    clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        #)
        self.encoder_hidden_states_proj = project(clip_embeddings_dim, 512, cross_attention_dim) #nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.text_encoder_hidden_states_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeddings, color_embeddings, text_encoder_outputs):
        prompt_embeds = text_encoder_outputs["text_embeds"]
        text_encoder_hidden_states = text_encoder_outputs["last_hidden_state"]

        """
        if do_classifier_free_guidance:
            # Add the classifier free guidance embeddings to the image embeddings
            image_embeddings_batch_size = image_embeddings.shape[0]
            classifier_free_guidance_embeddings = self.learned_classifier_free_guidance_embeddings.unsqueeze(0)
            classifier_free_guidance_embeddings = classifier_free_guidance_embeddings.expand(
                image_embeddings_batch_size, -1
            )
            image_embeddings = torch.cat([classifier_free_guidance_embeddings, image_embeddings], dim=0)
        """
        
        # The image embeddings batch size and the text embeddings batch size are equal
        assert image_embeddings.shape[0] == prompt_embeds.shape[0]

        batch_size = prompt_embeds.shape[0]

        # "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and
        # adding CLIP embeddings to the existing timestep embedding, ...
        time_projected_prompt_embeds = self.embedding_proj(prompt_embeds) #(B, )
        time_projected_image_embeddings = self.clip_image_embeddings_project_to_time_embeddings(image_embeddings) #(B, )
        time_projected_color_embeddings = self.color_embeddings_project_to_time_embeddings(color_embeddings) #(B, )
        additive_clip_time_embeddings = time_projected_image_embeddings + time_projected_color_embeddings + time_projected_prompt_embeds

        #NOTE: DONE!
        # ... and by projecting CLIP embeddings into four
        # extra tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder"
        # Project image embedding
        clip_extra_context_tokens_img = self.clip_extra_context_tokens_proj(image_embeddings) # (B, 2048)
        clip_extra_context_tokens_img = clip_extra_context_tokens_img.reshape(batch_size, self.clip_extra_context_tokens, -1) #(B, 4, 512)

        # Project color embedding
        clip_extra_context_tokens_color = self.color_extra_context_tokens_proj(color_embeddings) # (B, 2048)
        clip_extra_context_tokens_color = clip_extra_context_tokens_color.reshape(batch_size, self.clip_extra_context_tokens, -1) #(B, 4, 512)


        text_encoder_hidden_states = self.encoder_hidden_states_proj(text_encoder_hidden_states) # (B, 77, 512)
        text_encoder_hidden_states = self.text_encoder_hidden_states_norm(text_encoder_hidden_states)
        #text_encoder_hidden_states = text_encoder_hidden_states.permute(0, 2, 1)
        text_encoder_hidden_states = torch.cat([clip_extra_context_tokens_img, clip_extra_context_tokens_color, text_encoder_hidden_states], dim=1) # (B, 85, 512)

        return text_encoder_hidden_states, additive_clip_time_embeddings