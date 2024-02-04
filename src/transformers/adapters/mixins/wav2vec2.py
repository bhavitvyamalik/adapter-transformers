from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin


class Wav2Vec2EncoderLayerAdaptersMixin:
    """Adds adapters to the Encoder Layer module of Wav2Vec2."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class Wav2Vec2EncoderLayerStableLayerNormAdaptersMixin:
    """Adds adapters to the Encoder Layer Stable Layer Norm module of Wav2Vec2."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class Wav2Vec2ModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the Wav2Vec2 module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layers):
            yield i, layer


class Wav2Vec2ModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass