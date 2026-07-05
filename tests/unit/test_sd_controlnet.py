import torch
from diffusers import ControlNetModel

from sd.train_controlnet import swap_cond_channels


def _tiny_controlnet():
    # conditioning_embedding_out_channels has 3 entries -> 2 stride-2 steps in the tiny
    # hint-encoder net, so the hint must be 4x the latent's spatial size to match after
    # downsampling (mirrors SD1.5's real 8x/conditioning_embedding_out_channels=4-entry setup).
    cfg = dict(
        in_channels=4,
        conditioning_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        mid_block_type="UNetMidBlock2D",
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=4,
        conditioning_embedding_out_channels=(8, 16, 32),
    )
    return ControlNetModel.from_config(cfg)


def test_swap_cond_channels_accepts_2ch_hint():
    cn = _tiny_controlnet()
    old_conv = cn.controlnet_cond_embedding.conv_in
    swap_cond_channels(cn, n=2)
    new_conv = cn.controlnet_cond_embedding.conv_in
    assert new_conv.in_channels == 2
    assert new_conv.out_channels == old_conv.out_channels
    assert new_conv.kernel_size == old_conv.kernel_size

    sample = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([1, 5])
    encoder_hidden_states = torch.randn(2, 4, 32)
    hint = torch.randn(2, 2, 64, 64)
    down_res, mid_res = cn(
        sample, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=hint, return_dict=False
    )
    assert mid_res.shape[0] == 2
    assert all(d.shape[0] == 2 for d in down_res)


if __name__ == "__main__":
    test_swap_cond_channels_accepts_2ch_hint()
    print("ok")
