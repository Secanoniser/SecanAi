from transformers import PretrainedConfig

class NanoLLMConfig(PretrainedConfig):
    model_type = "nanollm"

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

if __name__ == "__main__":
    config = NanoLLMConfig()
    print("[+] NanoLLMConfig initialized successfully:")
    print(config)
