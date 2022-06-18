import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForSequenceClassification

import math

class RMTEncoderForSequenceClassification():
    def __init__(self, config=None, base_model=None, **kwargs):
        if config is not None:
            self.model = AutoModelForSequenceClassification(config, **kwargs)
        
        if base_model is not None:
            self.model = base_model


    def from_pretrained(from_pretrained, **kwargs):
        base_model = AutoModelForSequenceClassification.from_pretrained(from_pretrained, **kwargs)
        rmt = RMTEncoderForSequenceClassification(base_model=base_model)
        return rmt
        

    def set_params(self, model_attr='bert', input_size=None, input_seg_size=None, num_mem_tokens=0, bptt_depth=-1, 
                    pad_token_id=0, cls_token_id=101, sep_token_id=102):
        self.net = getattr(self.model, model_attr)
        self.input_size =  self.net.embeddings.position_embeddings.weight.shape[0] if input_size is None else input_size
        self.input_seg_size = input_seg_size

        self.bptt_depth = bptt_depth
        self.pad_token_id = pad_token_id
        self.cls_token = torch.tensor([cls_token_id])
        self.sep_token = torch.tensor([sep_token_id])
        self.num_mem_tokens = num_mem_tokens
        self.extend_word_embeddings()


    def set_memory(self, memory=None):
        if memory is None:
            mem_token_ids = self.mem_token_ids.to(device=self.device)
            memory = self.net.embeddings.word_embeddings(mem_token_ids)
        return memory
    
    def extend_word_embeddings(self):
        vocab_size = self.net.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + self.num_mem_tokens
        self.mem_token_ids = torch.arange(vocab_size, vocab_size + self.num_mem_tokens)
        self.net.resize_token_embeddings(extended_vocab_size)


    def __call__(self, input_ids, **kwargs):
        # print(kwargs)
        # np.save('kwargs.npy', kwargs, allow_pickle=True)
        # return
        memory = self.set_memory()
        
        segmented = self.pad_and_segment(input_ids)
        for seg_num, segment_data in enumerate(zip(*segmented)):
            input_ids, attention_mask, token_type_ids = segment_data
            if memory.ndim == 2:
                memory = memory.repeat(input_ids.shape[0], 1, 1)
            if (self.bptt_depth > -1) and (len(segmented) - seg_num > self.bptt_depth): 
                memory = memory.detach()

            inputs_embeds = self.net.embeddings.word_embeddings(input_ids)
            inputs_embeds[:, 1:1+self.num_mem_tokens] = memory

            seg_kwargs = dict(**kwargs)
            seg_kwargs['inputs_embeds'] = inputs_embeds
            seg_kwargs['attention_mask'] = attention_mask
            seg_kwargs['token_type_ids'] = token_type_ids
            
            out = self.model.forward(**seg_kwargs, output_hidden_states=True)
            memory = out.hidden_states[-1][:, :self.num_mem_tokens]

        return out

    def pad_and_segment(self, input_ids):
        
        sequence_len = input_ids.shape[1]
        
        input_seg_size = self.input_size if self.input_seg_size is None else self.input_seg_size
        input_seg_size = input_seg_size - self.num_mem_tokens - 3 
            
        n_segments = math.ceil(sequence_len / input_seg_size)
        print(f'sequence_len: {sequence_len}, input_seg_size: {input_seg_size}')

        augmented_inputs = []
        for input in input_ids:
            input = input[input != self.pad_token_id][1:-1]
            print(f'raw input: {input.shape}')

            seg_sep_inds = [0] + list(range(len(input), 0, -input_seg_size))[::-1] # chunk so that first segment has various size
            print(f'splitting to {seg_sep_inds}')
            input_segments = [input[s:e] for s, e in zip(seg_sep_inds, seg_sep_inds[1:])]

            def pad_add_special_tokens(tensor, seg_size):
                tensor = torch.cat([self.cls_token.to(device=self.device),
                                    self.mem_token_ids.to(device=self.device),
                                    self.sep_token.to(device=self.device),
                                    tensor.to(device=self.device),
                                    self.sep_token.to(device=self.device)])
                pad_size = seg_size - tensor.shape[0]
                if pad_size > 0:
                    tensor = F.pad(tensor, (0, pad_size))
                return tensor

            input_segments = [pad_add_special_tokens(t, self.input_size) for t in input_segments]
            print(f'got input segments {input_segments}, {[len(i) for i in input_segments]}')
            empty = torch.Tensor([]).int()
            empty_segments = [pad_add_special_tokens(empty, self.input_size) for i in range(n_segments - len(input_segments))]
            input_segments = empty_segments + input_segments
            print(f'padded input segments {input_segments}, {[len(i) for i in input_segments]}')

            augmented_input = torch.cat(input_segments)
            augmented_inputs.append(augmented_input)
            
        augmented_inputs = torch.stack(augmented_inputs)
        attention_mask = torch.ones_like(augmented_inputs)
        attention_mask[augmented_inputs == self.pad_token_id] = 0

        token_type_ids = torch.zeros_like(attention_mask)

        input_segments = torch.chunk(augmented_inputs, n_segments, dim=1)
        attention_mask = torch.chunk(attention_mask, n_segments, dim=1)
        token_type_ids = torch.chunk(token_type_ids, n_segments, dim=1)
    
        return input_segments, attention_mask, token_type_ids


    def to(self, device):
        self.model = self.model.to(device)
        
    
    def cuda(self):
        self.model.cuda()


    def __getattr__(self, attribute):
        return getattr(self.model, attribute)


    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def named_parameters(self, **kwargs):
        return self.model.named_parameters(**kwargs)