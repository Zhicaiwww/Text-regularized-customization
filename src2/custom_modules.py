from packaging import version
import torch
import torch.nn as nn
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
import sys
sys.path.append('/home/zhicai/poseVideo/custom-diffusion/')
from src2.parse_prompts import parse_prompt_attention

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
class ClassBias(nn.Module):
    def __init__(self,modifier_id, class_ids, shape):
        super().__init__()
        self.modifier_id = modifier_id
        self.class_ids = class_ids
        self.class_bias = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.class_len = len(class_ids)

class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []
        
class FrozenCLIPEmbedderWrapper(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, modifier_token, concept_classes = None, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,num_vectors_per_token=1,enable_emphasis=True,interpolated =False):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.modifier_token = modifier_token
        self.num_vectors_per_token = num_vectors_per_token
        self.chunk_length = 75
        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.id_end
        self.enable_emphasis = enable_emphasis
        self.concept_classes = concept_classes #["'<new1> dog'"]
        self.class_bias = True if concept_classes is not None else False
        self.interpolated = interpolated    
        if '+' in self.modifier_token:
            self.modifier_token = self.modifier_token.split('+')
        else:
            self.modifier_token = [self.modifier_token]
        # if class_bias:
            

        self.add_token()
        self.freeze()

    def add_token(self):
        self.modifier_token_id = []
        token_embeds1 = self.transformer.get_input_embeddings().weight.data
        for each_modifier_token in self.modifier_token:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token)
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token)
            self.modifier_token_id.append(modifier_token_id)

        self.transformer.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.transformer.get_input_embeddings().weight.data
        
        self.pad_embedding = token_embeds[self.id_pad]
        token_embeds[self.modifier_token_id[-1]] = torch.nn.Parameter(token_embeds[42170], requires_grad=True)
        if len(self.modifier_token) == 2:
            token_embeds[self.modifier_token_id[-2]] = torch.nn.Parameter(token_embeds[47629], requires_grad=True)
        if len(self.modifier_token) == 3:
            token_embeds[self.modifier_token_id[-3]] = torch.nn.Parameter(token_embeds[43514], requires_grad=True)
        
        if self.class_bias:
            for each_concept_class in self.concept_classes:
                class_modifier, class_name = each_concept_class.split(' ',1)
                print(class_name)
                assert class_modifier in self.modifier_token
                class_modifier_id = self.tokenize(class_modifier)[0]
                class_name_id = self.tokenize(class_name) 
                class_ids = torch.asarray(class_name_id)
                self.class_manager = ClassBias(class_modifier_id, class_ids, token_embeds[class_name_id].size())
                # self.modifier_concept_class_dict[class_modifier_id] = class_manager

    def custom_forward(self, hidden_states, tokens, multipliers):
        r"""
        Returns:
        """
        input_shape = hidden_states.size()
        bsz, seq_len = input_shape[:2]
        if self.interpolated:
            pad_hidden_state = self.pad_embedding.expand(bsz, seq_len, -1).detach().to(hidden_states.device)
            # calculate iterplation between original hidden state and pad hidden state using multipliers, iterplated =  original * multipliers + pad * (1-multipliers)
            hidden_states = hidden_states * multipliers.unsqueeze(-1) + pad_hidden_state * (1 - multipliers.unsqueeze(-1))
        else:
            original_mean = hidden_states.mean(dim=[1,2],keepdim=True)
            hidden_states = hidden_states * multipliers.unsqueeze(-1)
            new_mean = hidden_states.mean(dim=[1,2],keepdim=True)
            hidden_states = hidden_states * (original_mean / new_mean)
        if version.parse(transformers.__version__) >= version.parse('4.21'):
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
        else:
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len).to(
                hidden_states.device
            )

        encoder_outputs = self.transformer.text_model.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)
        # original_mean = last_hidden_state.mean(dim=[1,2],keepdim=True)
        # last_hidden_state = last_hidden_state * multipliers.unsqueeze(-1)
        # new_mean = last_hidden_state.mean(dim=[1,2],keepdim=True)
        # last_hidden_state = last_hidden_state * (original_mean / new_mean)
        return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = False

    def forward(self, texts):

        cache = {}
        batch_tokens = []
        batch_multipliers = []
        for line in texts:
            if line in cache:
                line_tokens,line_multipliers = cache[line]
            else:
                line_tokens, line_multipliers = self.tokenize_line(line)
                cache[line] = (line_tokens, line_multipliers)
            batch_tokens.append(line_tokens)
            batch_multipliers.append(line_multipliers)
        # n, 77
        tokens = torch.stack(batch_tokens).to(self.device)
        # n, 77
        multipliers = torch.stack(batch_multipliers).to(self.device)
        
        # import pdb; pdb.set_trace()

        if len(self.modifier_token) == 3:
            indices = ((tokens == self.modifier_token_id[-1]) | (tokens == self.modifier_token_id[-2]) | (tokens == self.modifier_token_id[-3]))*1
        elif len(self.modifier_token) == 2:
            indices = ((tokens == self.modifier_token_id[-1]) | (tokens == self.modifier_token_id[-2]))*1
        else:
            indices = (tokens == self.modifier_token_id[-1])*1

        indices = indices.unsqueeze(-1)

        input_shape = tokens.size()
        tokens = tokens.view(-1, input_shape[-1])

        hidden_states = self.transformer.text_model.embeddings(input_ids=tokens)
        hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states

        if self.class_bias:
            # for modifier_id, classmanager in self.modifier_concept_class_dict.items():
                modifier_id = self.class_manager.modifier_id
                idxs = torch.where(tokens == modifier_id)
                if len(idxs[0]) == 0:
                    pass
                    # break
                else:
                    for i in range(len(idxs[0])):
                        b,t = idxs[0][i], idxs[1][i]
                        if t + self.class_manager.class_len + 1 < 77 and tokens[b,t+1:t+1+self.class_manager.class_len].equal(self.class_manager.class_ids.to(self.device)):
                            hidden_states[b,t+1:t+1+self.class_manager.class_len] = hidden_states[b,t+1:t+1+self.class_manager.class_len] + self.class_manager.class_bias
        return hidden_states, tokens , multipliers

    def tokenize(self,texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        if self.enable_emphasis:
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunk = PromptChunk()
        token_count = 0
        is_last = False

        for tokens, (_, weight) in zip(tokenized, parsed):
            position = 0
            while position < len(tokens):
                token = tokens[position]
                if len(chunk.tokens) == self.chunk_length:
                    is_last = True
                    break
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

            if is_last:
                break

        to_add = self.chunk_length - len(chunk.tokens)
        if to_add > 0:
            chunk.tokens += [self.id_end] * to_add
            chunk.multipliers += [1.0] * to_add

        chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
        line_tokens = torch.asarray(chunk.tokens)
        line_multipliers = torch.asarray(chunk.multipliers)
        return line_tokens, line_multipliers
    
    def encode(self, texts):

        hidden_states, tokens, multipliers = self(texts)
        z = self.custom_forward(hidden_states, tokens, multipliers)

        # z[torch.arange(z.shape[0]), 0] = 0
        # z[torch.arange(z.shape[0]), [torch.where(tokens[i] == 49407,)[0][0] for i in range(len(tokens))]] = 0
        return z

    def encode_text(self, text):
        hidden_states, tokens, multipliers = self(text)
        z = self.custom_forward(hidden_states, tokens, multipliers)
        stc_z = z[torch.arange(z.shape[0]), [torch.where(tokens[i] == 49407,)[0][0] for i in range(len(tokens))]]
        return z, stc_z
        


if __name__ == "__main__":
    model = FrozenCLIPEmbedderWrapper(modifier_token='<new1>',concept_classes = ['<new1> cat dog'], device='cpu')
    model.encode(["hello (world:2) <new1> dog, <new1> cat dog"])
    model.encode(["hello (world:2)"])