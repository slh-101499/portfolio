# -*- coding: utf-8 -*-
"""
交互式问答入口：用自然语言问数据（PandasAI + 本地 Ollama）。

用法:
    python3 ask.py <csv路径> "哪个门店客单价最高？"
    python3 ask.py <csv路径> "画一张各门店类型平均交易金额的柱状图"

依赖：本机已启动 Ollama（ollama serve），且已 pull qwen2.5-coder:3b。
图表会存到 exports/charts/ 下。
"""
import sys
from pandasai_ollama import run_pandasai


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    csv_path, question = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5-coder:3b"
    agent, df = run_pandasai(csv_path, model=model)
    print(f"数据: {df.shape[0]:,} 行 × {df.shape[1]} 列 | 模型: {model}\n")
    print(f"❓ {question}\n")
    r = agent.chat(question)
    if hasattr(r, 'value'):
        print(f"→ {r.value}")
    else:
        print(f"→ {r}")


if __name__ == "__main__":
    main()
