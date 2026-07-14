import torch
from types import SimpleNamespace
from model_all import HNeRV


def make_args(repr_type='baseline', rank=0):
    return SimpleNamespace(
        embed='pe_1.25_8',
        repr_type=repr_type,
        nlrt_rank=rank,
        nlrt_h0=1,
        nlrt_w0=1,
        nlrt_c=16,
        nlrt_t_pe='1.25_8',
        nlrt_use_spatial_mlp=0,
        full_data_length=8,
        ks='0_3_3',
        num_blks='1_1',
        enc_strds=[],
        enc_dim='64_16',
        conv_type=['convnext', 'pshuffel'],
        norm='none',
        act='gelu',
        dec_strds=[2, 2],
        reduce=2.0,
        lower_width=4,
        fc_dim=16,
        fc_hw='2_2',
        out_bias='tanh',
    )


def run_case(repr_type, rank):
    args = make_args(repr_type=repr_type, rank=rank)
    model = HNeRV(args)
    model.train()
    norm_idx = torch.linspace(0, 1, steps=4)
    out, _, _ = model(norm_idx)
    loss = out.mean()
    if repr_type == 'nlrt':
        aux = model.nlrt_losses(norm_idx, out.device)
        loss = loss + aux['loss_temp'] + aux['loss_reg']
    loss.backward()
    return out.shape


if __name__ == '__main__':
    shape_base = run_case('baseline', 0)
    shape_nlrt = run_case('nlrt', 8)
    print(f'baseline shape: {shape_base}, nlrt shape: {shape_nlrt}')
    print('smoke_test_nlrt: PASS')
