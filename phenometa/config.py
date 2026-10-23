from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    latent_dim: int
    n_classes: int
    metacell_feature_dim: int | None = None
    init_scale: float = 2.0
    min_init_prototypes: int = 16
    max_init_prototypes: int = 2048
    top_k_ratio: float = 0.05
    min_top_k: int = 5
    max_top_k: int = 50
    refinement_tau: float = 0.05
    usage_alpha: float = 0.001
    convergence_tol: float = 1e-4
    max_refinement_iter: int = 100
    dead_eps: float = 1e-4
    merge_top_r: int = 20
    merge_threshold: float = 0.99
    merge_usage_ratio_threshold: float = 3.0
    max_outer_iter: int = 100
    fps_subset_size: int = 100000
    search_chunk_size: int = 4096
    gate_hidden_dim: int = 128
    bias_weight: float = 1.0
    use_assignment_gate: bool = True
    use_bias_in_refinement: bool = True
    use_bias_in_final_assignment: bool = True
    use_st_gumbel_final_assignment: bool = True
    use_interaction_pair_features: bool = False
    pooling: str = "size_aware_attention"
    predictor_encoder_dims: tuple[int, int] = (1024, 512)
    predictor_hidden_dim: int = 256
    predictor_classifier_dims: tuple[int, int] = (128, 64)
    predictor_encoder_dropout: float = 0.1
    pooling_hidden_dim: int = 128
    classifier_hidden_dim: int = 128  # backward-compatible, unused
    classifier_dropout: float = 0.1
    eps: float = 1e-12

    def to_dict(self):
        return asdict(self)
