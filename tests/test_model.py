import torch
import torch.nn.functional as F

from phenometa.config import ModelConfig
from phenometa.model import PhenoMetaModel


def test_phenotype_loss_reaches_assignment_gate():
    torch.manual_seed(7)
    config = ModelConfig(
        latent_dim=4,
        n_classes=2,
        metacell_feature_dim=6,
        min_init_prototypes=3,
        max_init_prototypes=3,
        min_top_k=2,
        max_top_k=2,
        max_refinement_iter=2,
        merge_threshold=1.1,
        predictor_encoder_dims=(8, 6),
        predictor_hidden_dim=5,
        predictor_classifier_dims=(4, 3),
        pooling_hidden_dim=4,
    )
    model = PhenoMetaModel(config).train()
    output = model(torch.randn(12, 4), torch.randn(12, 6), gumbel_tau=0.8)
    loss = F.cross_entropy(output["logits"][None], torch.tensor([1]))
    loss.backward()
    gradient = model.assignment_gate.output_layer.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
