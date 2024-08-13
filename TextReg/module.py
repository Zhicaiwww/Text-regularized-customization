import random
from typing import Optional, Tuple, Union

import torch
from transformers import CLIPTextModel
from transformers.models.clip.modeling_clip import (
    BaseModelOutputWithPooling,
    CLIPTextConfig,
    CLIPTextTransformer,
    _expand_mask,
)


def get_causal_attention_mask(bs, seq_len, dtype: type = torch.float16):

    mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


def get_identifier_masked_causal_attention_mask(
    bs,
    seq_len,
    identifier_indices: torch.Tensor,
    dtype: type = torch.float16,
    class_token_len: int = -1,
    mask_identifier_ratio: float = 0.5,
):

    if random.random() < mask_identifier_ratio:
        assert class_token_len >= 1
        for i, identifier_indice in zip(identifier_indices[0], identifier_indices[1]):
            mask = torch.empty(bs, seq_len, seq_len, dtype=dtype)
            mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            mask[i, :, identifier_indice, : max(identifier_indice, 1)] = torch.finfo(
                dtype
            ).min
            mask[i, :, identifier_indice + class_token_len + 1 :, identifier_indice] = (
                torch.finfo(dtype).min
            )
    else:
        mask = get_causal_attention_mask(bs, seq_len, dtype)
    return mask


class CLIPTiTextTransformer(CLIPTextTransformer):
    def __init__(
        self,
        config: CLIPTextConfig,
        mask_identifier_causal_attention: Optional[bool] = False,
        class_token_len: Optional[int] = -1,
        placeholder_token_id: Optional[int] = 49408,
        mask_identifier_ratio=0.5,
    ):
        super().__init__(config)
        self.mask_identifier_causal_attention = mask_identifier_causal_attention
        self.class_token_len = class_token_len
        self.placeholder_token_id = placeholder_token_id
        self.mask_identifier_ratio = mask_identifier_ratio

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_ids.size()

        if self.mask_identifier_causal_attention:
            identifier_indices = torch.where(input_ids == self.placeholder_token_id)
            causal_attention_mask = get_identifier_masked_causal_attention_mask(
                bsz,
                seq_len,
                identifier_indices,
                class_token_len=self.class_token_len,
                dtype=hidden_states.dtype,
                mask_identifier_ratio=self.mask_identifier_ratio,
            ).to(hidden_states.device)
        else:
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = get_causal_attention_mask(
                bsz, seq_len, dtype=hidden_states.dtype
            ).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=input_ids.device),
            [(row == 49407).nonzero().min() for row in input_ids],
        ]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTiTextModel(CLIPTextModel):
    config_class = CLIPTextConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config, **kwargs):
        super(CLIPTextModel, self).__init__(config)
        self.text_model = CLIPTiTextTransformer(config, **kwargs)
        self.post_init()
