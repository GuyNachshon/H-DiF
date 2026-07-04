from models.hdit import (
    GlobalAttentionSpec,
    GlobalTransformerLayer,
    ImageTransformerDenoiserModelV2,
    LevelSpec,
    MappingSpec,
    NoAttentionSpec,
    ShiftedWindowAttentionSpec,
)

_ATTN = {
    "global": lambda a: GlobalAttentionSpec(d_head=a["d_head"]),
    "shifted-window": lambda a: ShiftedWindowAttentionSpec(d_head=a["d_head"], window_size=a["window_size"]),
    "none": lambda a: NoAttentionSpec(),
}


def build_hdit(model_cfg: dict) -> ImageTransformerDenoiserModelV2:
    widths = model_cfg["widths"]
    depths = model_cfg["depths"]
    d_ffs = model_cfg["d_ffs"]
    self_attns = model_cfg["self_attns"]
    dropout = model_cfg["dropout"]
    patch_size = tuple(model_cfg["patch_size"])

    levels = [
        LevelSpec(depth=d, width=w, d_ff=f, self_attn=_ATTN[a["type"]](a), dropout=drop)
        for w, d, f, a, drop in zip(widths, depths, d_ffs, self_attns, dropout)
    ]
    m = model_cfg["mapping"]
    mapping = MappingSpec(depth=m["depth"], width=m["width"], d_ff=m["d_ff"], dropout=m["dropout"])

    model = ImageTransformerDenoiserModelV2(
        levels=levels,
        mapping=mapping,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        patch_size=patch_size,
    )

    for layer in model.mid_level:
        if isinstance(layer, GlobalTransformerLayer):
            layer.self_attn.use_cache = True

    # Bottleneck token count must stay small enough for full global attention (O(hw^2)).
    if "size" in model_cfg:
        side = model_cfg["size"] // patch_size[0] // 2 ** (len(widths) - 1)
        hw = side * side
        if hw > 1024:
            raise ValueError(f"bottleneck token count {hw} exceeds 1024; reduce size or add a level")

    return model
