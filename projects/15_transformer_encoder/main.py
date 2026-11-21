"""
Minimal Transformer encoder from scratch.

Forward-pass only

Mirrors 14_attention's narrative: visualize the real thing first (one real
protein sequence walked through every stage of the forward pass), then
verify the mechanics (shapes, softmax, LayerNorm, the residual add) on
synthetic data, then cross-check against PyTorch's own built-in modules.

Usage:
    python main.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_load_prep import VOCAB_SIZE, load_sequence, tokenize
from encoder import EncoderLayer, TransformerEncoder
from feed_forward import PositionwiseFeedForward
from multi_head_attention import MultiHeadAttention
from plotting import (
    plot_attention_heads,
    plot_attention_scores_heads,
    plot_embedding_stages,
    plot_ffn_pipeline,
    plot_head_detail,
    plot_representation_norms,
    plot_residual_effect,
)
from positional_encoding import sinusoidal_positional_encoding

FIG_DIR = Path(__file__).resolve().parent / "fig"



def build_embedding_demo(d_model) -> tuple[str, torch.Tensor, torch.Tensor]:
    '''
    step 1: embed a real sequence and add positional encoding, visualizing
    both stages and their sum
    '''

    # pick 1 sequence
    sequence = load_sequence(length=12, seed=0)
    
    ids = tokenize(sequence)  # (1, length)
    
    torch.manual_seed(6)
    
    embedding = nn.Embedding(VOCAB_SIZE, d_model)
    
    with torch.no_grad():
        token_emb = embedding(ids) * (d_model**0.5)
        pos_enc = sinusoidal_positional_encoding(len(sequence), d_model)  # (12, d_model)
        
        combined = token_emb + pos_enc.unsqueeze(0)

    plot_embedding_stages(
        token_emb[0].numpy(), 
        pos_enc.numpy(), 
        combined[0].numpy(), 
        sequence, FIG_DIR / "embedding_stages.png"
    )

    return sequence, token_emb, combined



def walk_through_layer(
    layer: EncoderLayer,
    rep: torch.Tensor,
    sequence: str,
    layer_idx: int,
) -> tuple[torch.Tensor, dict]:
    '''
    step 2: manually reproduce EncoderLayer.forward one piece at a time
    '''

    # get multihead attn layer; shape (1, 12, 5)
    mha = layer.self_attn 
    
    # --- multi-head attention sublayer ---
    # (1, H, L, d_k)
    Q = mha._split_heads(mha.W_q(rep))  
    K = mha._split_heads(mha.W_k(rep))
    V = mha._split_heads(mha.W_v(rep))

    scores = Q @ K.transpose(-2, -1) / math.sqrt(mha.d_k)  # (1, H, L, L)
    

    attn_out, attn_weights = mha(rep) 
    output = attn_weights @ V  # (1, H, L, d_k) -- per-head output, before W_o merges heads

    plot_head_detail(
        Q[0, 0].numpy(),
        K[0, 0].numpy(),
        V[0, 0].numpy(),
        output[0, 0].numpy(),
        sequence,
        FIG_DIR / f"head0_detail_layer{layer_idx}.png",
        title=f"Layer {layer_idx}, head 0: Q, K, V, output",
    )
    plot_attention_scores_heads(
        scores[0].numpy(),
        sequence,
        FIG_DIR / f"scores_heads_layer{layer_idx}.png",
        title=f"Layer {layer_idx} attention scores (pre-softmax, all heads)",
    )
    plot_attention_heads(
        attn_weights[0].numpy(),
        sequence,
        FIG_DIR / f"attn_heads_layer{layer_idx}.png",
        title=f"Layer {layer_idx} attention weights (post-softmax, all heads)",
    )

    x_after_attn = layer.norm1(rep + attn_out)
    plot_residual_effect(
        rep[0].numpy(),
        attn_out[0].numpy(),
        (rep + attn_out)[0].numpy(), # combine representation with attn_out
        x_after_attn[0].numpy(),
        sequence,
        FIG_DIR / f"residual_attn_layer{layer_idx}.png",
        title=f"Layer {layer_idx}: residual + LayerNorm (attention sublayer)",
    )

    # # --- feed-forward sublayer ---
    ffn = layer.feed_forward
    hidden = ffn.relu(ffn.linear1(x_after_attn))  # (1, L, d_ff)
    ff_out = ffn.linear2(hidden)  # (1, L, d_model)

    plot_ffn_pipeline(
        x_after_attn[0].numpy(),
        hidden[0].numpy(),
        ff_out[0].numpy(),
        sequence,
        FIG_DIR / f"ffn_pipeline_layer{layer_idx}.png",
        title=f"Layer {layer_idx}: feed-forward sublayer",
    )

    x_after_ffn = layer.norm2(x_after_attn + ff_out)
    plot_residual_effect(
        x_after_attn[0].numpy(),
        ff_out[0].numpy(),
        (x_after_attn + ff_out)[0].numpy(),
        x_after_ffn[0].numpy(),
        sequence,
        FIG_DIR / f"residual_ffn_layer{layer_idx}.png",
        title=f"Layer {layer_idx}: residual + LayerNorm (feed-forward sublayer)",
    )

    # the manual walk above must match the module's own forward exactly
    real_out, real_attn_weights, real_x_after_attn = layer(rep)
    assert torch.allclose(x_after_ffn, real_out, atol=1e-5)
    assert torch.allclose(x_after_attn, real_x_after_attn, atol=1e-5)
    assert torch.allclose(attn_weights, real_attn_weights, atol=1e-5)

    stage = {"attn_weights": attn_weights, "after_attn": x_after_attn, "after_ffn": x_after_ffn}
    
    return x_after_ffn, stage



def check_against_pytorch_mha() -> None:
    torch.manual_seed(4)
    d_model = 32
    num_heads = 4
    B = 2
    L = 9
    x = torch.randn(B, L, d_model)

    attn_scratch = MultiHeadAttention(d_model, num_heads)

    attn_reference = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    with torch.no_grad():
        attn_reference.in_proj_weight.copy_(
            torch.cat([
                attn_scratch.W_q.weight, 
                attn_scratch.W_k.weight, 
                attn_scratch.W_v.weight
            ], dim=0))
        attn_reference.in_proj_bias.copy_(
            torch.cat([
                attn_scratch.W_q.bias, 
                attn_scratch.W_k.bias, 
                attn_scratch.W_v.bias
            ], dim=0))
        attn_reference.out_proj.weight.copy_(attn_scratch.W_o.weight)
        attn_reference.out_proj.bias.copy_(attn_scratch.W_o.bias)

        out_scratch, _ = attn_scratch(x)
        out_ref, _ = attn_reference(x, x, x, need_weights=False)

    max_diff = (out_scratch - out_ref).abs().max().item()
    assert torch.allclose(out_scratch, out_ref, atol=1e-4), max_diff

    print(f"[cross-check: multi-head attention] matches torch.nn.MultiheadAttention (max diff {max_diff:.2e}).")



def check_against_pytorch_encoder_layer() -> None:
    torch.manual_seed(5)

    d_model = 32
    num_heads = 4
    d_ff = 64
    B = 2
    L = 9

    x = torch.randn(B, L, d_model)

    scratch = EncoderLayer(d_model, num_heads, d_ff)
    
    reference = nn.TransformerEncoderLayer(
        d_model, 
        num_heads, 
        dim_feedforward=d_ff, 
        dropout=0.0, 
        activation="relu", 
        batch_first=True, 
        norm_first=False,
    )
    reference.eval()
    with torch.no_grad():
        mha = scratch.self_attn
        reference.self_attn.in_proj_weight.copy_(
            torch.cat([
                mha.W_q.weight, 
                mha.W_k.weight, 
                mha.W_v.weight
            ], dim=0))
        reference.self_attn.in_proj_bias.copy_(
            torch.cat([
                mha.W_q.bias, 
                mha.W_k.bias, 
                mha.W_v.bias
            ], dim=0))
        reference.self_attn.out_proj.weight.copy_(mha.W_o.weight)
        reference.self_attn.out_proj.bias.copy_(mha.W_o.bias)

        reference.linear1.weight.copy_(scratch.feed_forward.linear1.weight)
        reference.linear1.bias.copy_(scratch.feed_forward.linear1.bias)
        reference.linear2.weight.copy_(scratch.feed_forward.linear2.weight)
        reference.linear2.bias.copy_(scratch.feed_forward.linear2.bias)

        reference.norm1.weight.copy_(scratch.norm1.weight)
        reference.norm1.bias.copy_(scratch.norm1.bias)
        reference.norm2.weight.copy_(scratch.norm2.weight)
        reference.norm2.bias.copy_(scratch.norm2.bias)

        out_scratch, _, _ = scratch(x)
        out_ref = reference(x)

    max_diff = (out_scratch - out_ref).abs().max().item()
    assert torch.allclose(out_scratch, out_ref, atol=1e-4), max_diff
    print(f"[cross-check: encoder layer] matches torch.nn.TransformerEncoderLayer (max diff {max_diff:.2e}).")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    d_model = 8 # dim of each token's representation
    num_heads = 4 
    d_ff = 12 # FFN hidden layers
    num_layers = 2 # n encoder layers to stack

    # 1. embedding + positional encoding
        # shape (1, 12, 32)
    sequence, token_emb, rep = build_embedding_demo(d_model)
    
    # identify a particular stage: embedding, pos_encoding
    # this is for later visualization
    stage_names = ["token_emb", "+pos_enc"]
    representations = [token_emb[0], rep[0]]

    
    # 2. walk through each encoder layer
    # init the encoder, with shape (1, 12, 32)
    encoder = TransformerEncoder(
        num_layers, 
        d_model, 
        num_heads, 
        d_ff
    )
    
    with torch.no_grad():
        for i, layer in enumerate(encoder.layers):
            rep, stage = walk_through_layer(
                layer, # EncoderLayer
                rep,  
                sequence,  
                i,
            )
            stage_names += [
                f"layer{i}_after_attn", 
                f"layer{i}_after_ffn"
            ]
            representations += [
                stage["after_attn"][0], 
                stage["after_ffn"][0]
            ]

    # see how each aa representation changes through the Transformer
    # use L2 norm as a simply way to check that LayerNorm is working
    norms = np.stack(
        [rep.norm(dim=-1).numpy() for rep in representations]
    )
    
    plot_representation_norms(
        stage_names, 
        norms, 
        d_model, 
        FIG_DIR / "representation_norms.png"
    )

    # 3. cross-check against pytorch's own built-in modules
    check_against_pytorch_mha()
    check_against_pytorch_encoder_layer()


if __name__ == "__main__":
    main()
