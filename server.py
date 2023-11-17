import torch
import torch.nn as nn
from functools import reduce
import math
from operator import mul
from torch.nn.modules.utils import _pair

class PromptGenerator(nn.Module):

    def __init__(self, num_clients, config, hidden_dims=768,):
        super().__init__()
        prompt_dim = hidden_dims
        self.prompt_proj = nn.Identity()
        patch_size = (config.DATA.CROPSIZE, config.DATA.CROPSIZE)
        num_tokens = config.MODEL.PROMPT.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.client_agnostic_prompt_basis = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_dim))
        
        # xavier_uniform initialization
        nn.init.uniform_(self.client_agnostic_prompt_basis.data, -val, val)

        # Not use Deep
        assert config.MODEL.PROMPT.DEEP == False
        # if self.prompt_config.DEEP:  # noqa

        #     total_d_layer = config.transformer["num_layers"]-1
        #     self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #         total_d_layer, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        
        # initiate prompt
        self.num_clients = num_clients
        self.client_descriptors = nn.Parameter(torch.zeros(1, num_clients, prompt_dim))
        nn.init.uniform_(self.client_descriptors.data, -val, val)

        # initiate hed, w_q, w_k, w_v
        # TODO
        self.num_heads = 12
        self.attention_head_dims = int(hidden_dims / self.num_heads)
        self.all_head_dims = self.num_heads * self.attention_head_dims
        self.w_q = nn.Linear(prompt_dim, self.all_head_dims)
        self.w_k = nn.Linear(prompt_dim, self.all_head_dims)
        self.w_v = nn.Linear(prompt_dim, self.all_head_dims)
        self.out = nn.Linear(hidden_dims, hidden_dims)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_dims)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self):
        # b num_clients num_tokens d*h
        mixed_q_layer = self.w_q(self.client_descriptors) # 1, num_client, num_heads * attention_head_dims
        mixed_k_layer = self.w_k(self.client_agnostic_prompt_basis) # 1, num_tokens, num_heads * attention_head_dims
        mixed_v_layer = self.w_v(self.client_agnostic_prompt_basis)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        attention_scores = torch.matmul(q_layer,k_layer.transpose(-1, -2)) # 1, num_head, num_clients, num_tokens
        attention_scores = attention_scores / math.sqrt(self.attention_head_dims)

        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs,v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dims,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 1 num_clients dims
        attention_output = self.out(context_layer)
        # 1 num_tokens dims
        client_specific_prompts = list()
        for i in range(self.num_clients):
            client_specific_prompts.append(self.client_agnostic_prompt_basis + attention_output[0,i,:])
        client_specific_prompts = torch.cat(client_specific_prompts, dim = 1)
        return client_specific_prompts
