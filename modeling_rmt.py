import math
import torch
import torch.nn.functional as F


class RMTBaseModel(torch.nn.Module):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.set_params(**rmt_kwargs)

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens, tokenizer)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1

    def set_memory(self, input_shape):
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.special_token_ids = [tokenizer.pad_token_id]
        for token in ['cls_token', 'sep_token', 'eos_token', 'bos_token']:
            token_id = getattr(tokenizer, f'{token}_id')
            if token_id is not None:
                self.register_buffer(token, torch.tensor([token_id]))
                self.special_token_ids.append(token_id)
            else:
                setattr(self, token, None)

    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)

        if hasattr(self.model.base_model, 'embeddings'): # enc-only
            self.model.embeddings = self.model.base_model.embeddings.word_embeddings
        elif hasattr(self.model.encoder, 'embed_tokens'): # enc-dec
            self.model.embeddings = self.model.encoder.embed_tokens
        else:
            raise NotImplementedError

    def forward(self, **kwargs):
       raise NotImplementedError

    def pad_and_segment(self, input_ids):
        segmented_batch = []
        for seq in input_ids:
            drop_mask = sum([seq == t for t in self.special_token_ids])
            seq = seq[(1 - drop_mask).bool()]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]

            align = self.rmt_config.get('segment_alignment')
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]

            # add empty segment markers if needed
            n_empty_segments = self.rmt_config['max_n_segments'] - len(input_segments)
            input_segments = [None] * n_empty_segments + input_segments

            segmented_batch.append(input_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch] \
                            for seg_num in range(self.rmt_config['max_n_segments'])]
        return segmented_batch

    def pad_add_special_tokens(self, **kwargs):
        raise NotImplementedError

    def prepare_kwargs(self, segment_input_ids, kwargs):
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask

        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.embeddings(input_ids)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if seg_kwargs.get('labels') is not None:
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        if seg_kwargs.get('token_type_ids') is not None:
            seg_kwargs['token_type_ids'] = self.get_token_type_ids(input_ids)
        seg_kwargs['output_hidden_states'] = True

        return seg_kwargs, non_empty_mask

    def process_outputs(self, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        segment_keys = ['loss']
        if output_attentions:
            segment_keys.append('attentions')
        if output_hidden_states:
            segment_keys.append('hidden_states')

        extracted = {}
        for seg_num, out in enumerate(model_outputs):
            for key, value in out.items():
                if any([sk in key for sk in segment_keys]):
                    extracted[f'{key}_{seg_num}'] = value

        if self.rmt_config['sum_loss']:
            losses = [out['loss'] for out in model_outputs]
            extracted['loss'] = torch.stack(losses).mean(dim=0)

        for key, value in extracted.items():
            rmt_out[key] = value

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        return rmt_out

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask


class RMTEncoderForMaskedLM(RMTBaseModel):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__(base_model, **rmt_kwargs)
        self.rmt_config['sum_loss'] = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'encoder_hidden_states': encoder_hidden_states, 'encoder_attention_mask': encoder_attention_mask,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict
                  }
        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids, labels)

        base_model_outputs = []
        for seg_num, segment in enumerate(zip(*segmented)):
            if self.rmt_config['bptt_depth'] > -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            out['seg_kwargs'] = seg_kwargs
            base_model_outputs.append(out)

        out = self.process_outputs(input_ids, base_model_outputs, output_attentions, output_hidden_states)

        return out

    def prepare_kwargs(self, segment, kwargs):
        segment_input_ids, segment_labels = segment
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask

        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.embeddings(input_ids)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        if seg_kwargs.get('token_type_ids') is not None:
            seg_kwargs['token_type_ids'] = self.get_token_type_ids(input_ids)
        seg_kwargs['output_hidden_states'] = True
        if seg_kwargs['labels'] is not None:
            seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])

        return seg_kwargs, non_empty_mask

    def process_outputs(self, input_ids, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        bs, seq_len = input_ids.shape

        losses = []
        logits = []
        labels_segm = []
        for out in model_outputs:
            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            labels_segm += [out['seg_kwargs']['labels']]

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        for i, l in enumerate(losses):
            rmt_out[f'loss_{i}'] = l.mean()

        # aggregate losses from all segments
        rmt_out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, bs - labels_segm[i].shape[0]), value=-100)

        rmt_out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        rmt_out['logits_segm'] = [logits]
        rmt_out['labels_segm'] = [labels_segm]

        return rmt_out

    def pad_and_segment(self, input_ids, labels=None):
        segmented_batch = []
        segmented_batch_labels = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels

        for seq, labels in zip(input_ids, batch_labels):
            content_tokens_mask = (seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())
            seq = seq[content_tokens_mask]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels is not None:
                labels = labels[content_tokens_mask]
                labels = labels[:self.segment_size * self.rmt_config['max_n_segments']]

            # n_seg = math.ceil(len(seq) / self.segment_size)
            # input_segments = torch.chunk(seq, n_seg)
            align = self.rmt_config.get('segment_alignment')
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]
            # add empty segment markers if needed
            n_empty_segments = self.rmt_config['max_n_segments'] - len(input_segments)
            input_segments = [None] * n_empty_segments + input_segments
            segmented_batch.append(input_segments)

            if labels is not None:
                labels_segments = [labels[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels') for t in labels_segments]
                labels_segments = [None] * n_empty_segments + labels_segments
                segmented_batch_labels.append(labels_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch]
                           for seg_num in range(self.rmt_config['max_n_segments'])]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.rmt_config['max_n_segments'])]

        return segmented_batch, segmented_batch_labels

    def pad_add_special_tokens(self, tensor, segment_size, add_to='inputs'):
        input_elements = []
        pad_value = 0
        if add_to == 'inputs':
            pad_value = self.pad_token_id
            input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        elif add_to == 'labels':
            pad_value = -100
            masked_labels = torch.ones((1), device=tensor.device, dtype=tensor.dtype) * pad_value
            input_elements += [masked_labels, masked_labels.repeat(self.num_mem_tokens), masked_labels, tensor, masked_labels]

        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=pad_value)
        return tensor
