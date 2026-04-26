from tollm.config import ModelConfig


def test_model_config_defaults() -> None:
    c = ModelConfig()
    assert c.hidden_size == 4096
    assert c.torch_dtype == "bfloat16"
