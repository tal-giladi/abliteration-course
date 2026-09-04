MAPPING: dict[str, str] = {
    "Model.abliterate": (
        "merge_into_model() in lib/ablate.py — same job (write a rank-1 correction into "
        "o_proj/down_proj for every layer), generalized over dtype, quantization, and "
        "discovered components instead of two hardcoded ones."
    ),
    "AbliterationParameters": (
        "The weights argument to merge_into_model() — a flat per-layer list in the toy "
        "course, replaced by a 4-parameter triangular kernel (max_weight, "
        "max_weight_position, min_weight, min_weight_distance) that generates that list."
    ),
    "get_abliterable_components": (
        "The hardcoded self_attn.o_proj / mlp.down_proj lookups inside merge_into_model — "
        "the real version discovers component names per architecture instead of assuming "
        "them, so it works on models this course's fixed Qwen3-0.6B target never has to "
        "handle."
    ),
    "LoraConfig": (
        "lora_rank1_factors() in lib/ablate.py — same rank-1 factorization, but installed "
        "as a real peft adapter on the model instead of returned as plain A/B tensors, so "
        "it can be reset to identity between search trials."
    ),
    "lora_A / lora_B assignment in Model.abliterate": (
        "lora_rank1_factors()'s A = (d @ W).unsqueeze(0) and B = (-d).unsqueeze(1) — "
        "identical formula, with B additionally scaled by this layer's strength-curve "
        "weight instead of always being applied at full strength."
    ),
    "residual_directions in main.py::run": (
        "direction.compute_direction(mean_a, mean_b) — same normalized difference of "
        "means, computed once before the search loop instead of once per trial."
    ),
    "Model.get_residuals_mean": (
        "direction.get_residuals_mean() — same name, same job (average last-token "
        "residuals over a prompt set); the real version streams the sum in float64 to "
        "avoid holding every prompt's full residual tensor in memory at once."
    ),
    "W = W.to(torch.float32) in Model.abliterate": (
        "No toy equivalent needed — lib/common.py hardcodes DTYPE = torch.float32 for "
        "every tensor in this course, so there's never a lower-precision weight to "
        "upcast before doing the projection math, or downcast afterward."
    ),
}
