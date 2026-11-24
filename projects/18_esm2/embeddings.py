from __future__ import annotations

import torch
from transformers import AutoModel
from transformers import AutoTokenizer

CHECKPOINT = "biohub/esm2_t6_8M_UR50D" # small 8M-parameter checkpoint

# loaded pre-trained model 
_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT) # create tokenizer, load vocab and config
_model = AutoModel.from_pretrained(CHECKPOINT)
_model.eval()

HIDDEN_SIZE = _model.config.hidden_size


def tokenize(
    sequence: str,
):
    """
    Turns a raw amino-acid string into the model's input ids.

    `<cls>` and `<eos>` get added automatically; 
    every standard amino acid maps to its own single token.
    """
    return _tokenizer(
        sequence,
        return_tensors="pt", # return as pytorch tensor
    )


def token_strings(
    sequence: str,
) -> list[str]:
    """
    The literal token strings (including `<cls>`/`<eos>`) for one sequence
    """
    encoded = tokenize(sequence)
    
    return _tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])


@torch.no_grad()
def residue_embeddings(
    sequence: str,
) -> torch.Tensor:
    """
    Runs ESM-2 encoder on one sequence.
    """
    encoded = tokenize(sequence)
    output = _model(**encoded) # pass tokenized seq through the encoder
    return output.last_hidden_state[0]


def noncontextual_embeddings(
    sequence: str,
) -> torch.Tensor:
    """
    bypass ESM-2 encoder, and returns the first layer before attention
    
    """
    encoded = tokenize(sequence)
    with torch.no_grad():
        return _model.embeddings.word_embeddings(encoded["input_ids"][0])


def sequence_embedding(
    sequence: str,
) -> torch.Tensor:

    output = residue_embeddings(sequence)
    output = output[1:-1]  # drop <cls> and <eos>; shape(len_seq, 320)

    # average the representation to a single number for each pos in a sequence
    # as a way to represent the whole sequence
    out = output.mean(dim=0)  # shape (320)
    
    return out
