import torch.nn as nn
from torchinfo import summary

import src.yaku.exp1.config as config


class DNN(nn.Module):
    """
    DNN
    """

    def __init__(
        self,
        *,
        input_dim,
        hidden_layers,
        output_dim,
    ):
        super().__init__()

        self.input_dim = input_dim

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module,
        weight_init_val=1.0,
        bias_init_val=0.0,
    ):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, bias_init_val)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, weight_init_val)
            nn.init.constant_(module.bias, bias_init_val)

    def forward(self, x):
        """
        Forward pass
        """
        _, input_dim = x.shape

        assert (
            input_dim == self.input_dim
        ), f"Actual input dim ({input_dim}) must be equal to expected input dim ({self.input_dim})"

        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    model = DNN(
        input_dim=config.INPUT_DIM,
        hidden_layers=config.HIDDEN_LAYERS,
        output_dim=config.OUTPUT_DIM,
    )

    summary(model, input_size=(config.LEARNING_BATCH_SIZE, config.INPUT_DIM))
