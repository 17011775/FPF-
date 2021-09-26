import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder 
from .generator import Generator
from .module import VarianceAdaptor, PositionalEncodings

class FastPitchFormant(nn.Module):
    """
    FastPitchFormant : Source-filter based Decomposed Modeling
    for Speech Synthesis, 2021 
    """
    def __init__(self, param: FastPitchFormantParameters):
        super(FastPitchFormant, self).__init__()
        self.param = param 

        self.encoder = Encoder(param) 
        self.pe = PositionalEncodings(param)
        self.variance_adaptor = VarianceAdaptor(param)

        self.formant_generator = Generator(param)
        self.excitation_generator = Generator(param)
        
        self.speaker_embedding = nn.Embedding(
            param.speaker_num, param.spk_embedding
        )
        self.decoder = Decoder(param)

    def forward(self, inputs: InputBlock) -> OutputBlock:
        text_mask = get_mask(inputs.text_length)
        spk_emb = self.speaker_embedding(inputs.speaker_id)
        
        encodings = self.encoder(inputs.text)

        (
            h,
            p,
            pred_pitches,
            log_durations,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            encodings, 
            spk_emb,
            text_mask,
            mel_mask,
            inputs.max_mel_len,
            inputs.pitchs,
            inputs.durations,
            inputs.p_control,
            inputs.d_control,
            )
        
        formant = self.formant_generator(h, mel_masks)
        excitation = self.excitation_generator(p, mel_mask, query = h)

        mel = self.decoder(formant, excitation, mel_mask)

        return OutputBlock()
        
    def get_mask(lengths, max_len=None):
        """Generate mask 
        Args:
            length: torch.Tensor, [B], lengths.
            max_len: Optional[int], maximum length.
        Returns:
            torch.Tensor, [B, max_len], mask.
        """ 
        batch_size = lengths.size(0)
        if max_len is None: 
            max_len = lengths.max()
        # [max_len]
        ids = torch.arange(0, max_len).to(lengths.device)
        # [B, max_len]
        mask = ids >= lengths.unsqueeze(1).expand(batch_size, -1).to(torch.float32)

        return mask