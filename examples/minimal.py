"""最小示例：可 pip install -e . 后运行。"""

from tollm import LLM, SamplingParams


def main() -> None:
    llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
    p = SamplingParams(temperature=0.0, max_tokens=8)
    out = llm.generate(["Hello, TOLlm."], p)
    print(out[0]["text"])


if __name__ == "__main__":
    main()
