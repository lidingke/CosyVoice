import torch
from cosyvoice.llm import TransformerLLM
import torch.nn.functional as F
from typing import Any, Dict,List, Tuple, Union
# 从环境变量里获取"VERSION"
import pickle
import sys
import os
VERSION = os.getenv("VERSION", "ORIGIN")

class Cache:
    llm_input: torch.Tensor
    cache_offset: int
    out_tokens: List
    is_infer_short: bool
    min_len: int

class TransformerLLMInfer(TransformerLLM):
    def __init__(self,save_temp=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()
        self.cache = Cache()
        self.beam_size: int = 1,
        self.sampling: int = 25,
        self.device = kwargs.get('device', 'cpu')
        self.save_temp = save_temp
        
    @torch.inference_mode()
    def inference_prefill_stage(self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            dtype: torch.dtype = torch.float32,
            ):
        device = text.device
        if self.save_temp:
            with open(f'{VERSION}_prefill_input.pkl', 'wb') as f:
                inputdata = {
                    "text": text.detach().cpu().numpy(),
                    "text_len": text_len.detach().cpu().numpy(),
                    "prompt_text": prompt_text.detach().cpu().numpy(),
                    "prompt_text_len": prompt_text_len.detach().cpu().numpy(),
                    "prompt_speech_token": prompt_speech_token.detach().cpu().numpy(),
                    "prompt_speech_token_len": prompt_speech_token_len.detach().cpu().numpy(),
                    "embedding": embedding.detach().cpu().numpy(),
                }
                pickle.dump(inputdata, f)
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        self.cache.is_infer_short = True if text_len < 40 else False
        max_seq = self.max_seq_short if self.cache.is_infer_short else self.max_seq_long

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1).to(dtype)

        # 4. cal min/max_length
        self.cache.min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        max_len = min(max_len, max_seq - lm_input.size(1)) # 生成token + prompt 总长度不超过 max_seq

        # 5. step by step decode
        self.cache.out_tokens = []
        offset = 0
        cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device)
        self.cache.cache_offset = 0

        # 6. first token
        y_pred = self.llm.inference_prefill(
            lm_input, offset=0,
            cache_offset=self.cache.cache_offset,
            att_mask=torch.tril(torch.ones((1, lm_input.shape[1], max_seq), device=lm_input.device)).to(torch.bool),
            fix_shape=True,
            is_infer_short=self.cache.is_infer_short,
        )
        self.cache.cache_offset = self.cache.cache_offset + lm_input.size(1)
        logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        top_ids = self.sampling_ids(logp.squeeze(dim=0), self.sampling, self.beam_size, ignore_eos=True).item()
        self.out_tokens.append(top_ids)
        self.cache.lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        if self.save_temp:
            with open(f'{VERSION}_prefill_output.pkl', 'wb') as f:
                outputdata = {
                    "out_tokens": self.cache.out_tokens,
                    "lm_input": self.cache.lm_input.detach().cpu().numpy(),
                    "cache_offset": self.cache.cache_offset,
                }
                pickle.dump(outputdata, f)
        return max_len

    @torch.inference_mode()
    def inference_decode_stage(self,
                               index:int,
                               ):
        y_pred = self.llm.inference_decode_step(
                    lm_input, offset=0,
                    cache_offset=self.cache.cache_offset,
                    att_mask=self.attn_mask_short[None, None, self.cache.cache_offset] if self.cache.is_infer_short else self.attn_mask_long[None, None, self.cache.cache_offset],
                    is_infer_short = self.cache.is_infer_short,
                )

        self.cache.cache_offset = self.cache.cache_offset + lm_input.size(1)
        logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        top_ids = self.sampling_ids(logp.squeeze(dim=0), self.sampling, self.beam_size, ignore_eos=True if index < self.cache.min_len else False).item()
        # if top_ids == self.speech_token_size:
            # return torch.tensor([[-1]], dtype=torch.int64, device=device)
        # in stream mode, yield token one by one
        # if stream is True:
            # yield torch.tensor([[top_ids]], dtype=torch.int64, device=device)
        self.cache.out_tokens.append(top_ids)
        offset += lm_input.size(1)
        lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        return torch.tensor([[top_ids]], dtype=torch.int64, device=self.device)
    
    @torch.inference_mode()
    def inference(self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        beam_size=1,
        sampling=25,
        max_token_text_ratio=30,
        min_token_text_ratio=3,
        stream=True
        ):
        max_len = self.inference_prefill_stage(
            text,
            text_len,
            prompt_text,
            prompt_text_len,
            prompt_speech_token,
            prompt_speech_token_len,
            embedding,
            max_token_text_ratio,
            min_token_text_ratio,
        )
        yield self.cache.out_tokens[0]
        for i in range(1, max_len):
            token = self.inference_decode_stage(i)
            if token == self.speech_token_size:
                break
            yield token
        