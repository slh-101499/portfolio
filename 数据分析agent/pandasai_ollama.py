# -*- coding: utf-8 -*-
"""
PandasAI 3.0 的 Ollama 适配器：用本地 Ollama 模型作为 PandasAI 的 LLM 后端。

pandasai 3.0 把 LLM 供应商拆出去了，只留 base 抽象类，所以需要自己实现 call()。
这里接 Ollama 的 /api/chat 接口，走 OpenAI 兼容的 chat 格式。

用法:
    from pandasai_ollama import OllamaLLM, run_pandasai
    agent = run_pandasai("xxx.csv", model="qwen2.5-coder:3b")
    print(agent.chat("哪个门店客单价最高？"))
"""
import requests
from pandasai.llm.base import LLM
from pandasai.agent import Agent
from pandasai.config import Config


class OllamaLLM(LLM):
    def __init__(self, model: str = "qwen2.5-coder:3b", host: str = "http://localhost:11434",
                 temperature: float = 0.0):
        super().__init__()
        self.model = model
        self.host = host
        self.temperature = temperature
        self._type = f"ollama-{model}"

    @property
    def type(self) -> str:
        return self._type

    def call(self, instruction, context=None) -> str:
        self.last_prompt = instruction.to_string()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.last_prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


def run_pandasai(csv_path: str, model: str = "qwen2.5-coder:3b",
                 host: str = "http://localhost:11434", sample_rows: int = None):
    """加载 CSV 并返回配置好 Ollama 的 PandasAI Agent。"""
    import pandas as pd
    from pandasai import DataFrame

    df = pd.read_csv(csv_path, skipinitialspace=True)
    if sample_rows:
        df = df.sample(sample_rows, random_state=42)
    pdf = DataFrame(df)
    config = Config(llm=OllamaLLM(model=model, host=host))
    return Agent([pdf], config=config), df


if __name__ == "__main__":
    import sys
    agent, df = run_pandasai(sys.argv[1] if len(sys.argv) > 1 else "synthetic_retail_20250901.csv")
    for q in [
        "这个数据集有多少行多少列？",
        "哪个门店类型(store_type)的平均交易金额(transaction_amount)最高？",
    ]:
        print(f"\n>>> {q}")
        print(agent.chat(q))
