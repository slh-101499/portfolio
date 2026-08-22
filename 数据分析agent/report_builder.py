# -*- coding: utf-8 -*-
"""
数据分析报告渲染器：把「分析 JSON + 原始 CSV」变成一份自包含的交互式 HTML 报告。

- 图表用 Plotly（可悬停、缩放、tooltip），中文原生支持（浏览器渲染）。
- 单文件自包含：plotly.js 一次性内嵌，无需联网即可打开/分享。
- 与 skill 的 JSON 输出（findings / answers / insights）对接。

用法:
    from report_builder import html_report
    html = html_report(df, findings, answers, insights, figure_builder)
    open("xxx_report.html","w").write(html)
"""
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

# 统一配色
C_PRIMARY = "#4f46e5"   # 主色 indigo
C_ACCENT  = "#f59e0b"   # 强调 amber
C_GREEN   = "#10b981"
C_RED     = "#ef4444"
C_GRAY    = "#6b7280"

FONT = "PingFang SC, -apple-system, Arial, sans-serif"

# 字段英文名 → 中文名（未收录的字段原样显示）
FIELD_ZH = {
    "transaction_id": "交易ID", "transaction_date": "交易日期", "age": "年龄", "gender": "性别",
    "annual_income": "年收入", "store_type": "门店类型", "product_category": "商品品类",
    "basket_size": "客单件数", "avg_item_price": "平均单价", "transaction_amount": "交易金额",
    "visit_frequency_monthly": "月到店频次", "loyalty_member": "忠诚会员", "loyalty_points": "忠诚积分",
    "online_purchase_pct": "线上购买占比", "return_rate": "退货率",
    "discount_sensitivity": "折扣敏感度", "satisfaction_score": "满意度评分",
    "payment_method": "支付方式", "days_since_last_visit": "距上次到访天数",
    "referral_source": "获客渠道", "customer_segment": "客户分层", "customer_lifetime_value": "客户终身价值",
    "workclass": "工作类型", "fnlwgt": "抽样权重", "education": "教育程度", "education-num": "教育年限",
    "marital-status": "婚姻状况", "occupation": "职业", "relationship": "家庭关系", "race": "种族",
    "sex": "性别", "capital-gain": "资本利得", "capital-loss": "资本损失",
    "hours-per-week": "周工时", "native-country": "国籍", "income": "收入等级",
}


def _zh(field_str):
    """把 'a/b/c' 形式的英文字段名翻译成中文，未收录的原样保留。"""
    return " / ".join(FIELD_ZH.get(p.strip(), p.strip()) for p in str(field_str).split("/"))


def _zh_wrap(field_str, per=3):
    """翻译字段名，每 `per` 个字段换行，避免多字段把「字段/类型」列撑太宽。"""
    zh = [FIELD_ZH.get(p.strip(), p.strip()) for p in str(field_str).split("/")]
    return "<br>".join(" / ".join(zh[i:i + per]) for i in range(0, len(zh), per))


# 正文里需要中文化的英文术语（只收录复合名/专有名词，避免 "age/sex" 这类短词误伤其他单词）
TEXT_ZH = {
    "customer_lifetime_value": "客户终身价值", "visit_frequency_monthly": "月到店频次",
    "discount_sensitivity": "折扣敏感度", "satisfaction_score": "满意度",
    "transaction_amount": "交易金额", "days_since_last_visit": "距上次到访天数",
    "online_purchase_pct": "线上购买占比", "avg_item_price": "平均单价",
    "loyalty_points": "忠诚积分", "annual_income": "年收入", "basket_size": "客单件数",
    "return_rate": "退货率", "loyalty_member": "忠诚会员", "payment_method": "支付方式",
    "product_category": "商品品类", "store_type": "门店类型", "customer_segment": "客户分层",
    "LTV": "客户终身价值", "Pareto": "帕累托", "segment": "客群",
    "Discount": "折扣店", "Luxury": "奢侈店", "Premium": "高端店", "Standard": "标准店",
    "Mobile": "移动支付", "Credit": "信用卡", "Debit": "借记卡", "Cash": "现金",
}
_TEXT_ZH_ITEMS = sorted(TEXT_ZH.items(), key=lambda kv: -len(kv[0]))


def _zh_text(s):
    """把正文里的英文字段名/取值尽量替换成中文（最长优先，避免部分匹配）。"""
    if not s:
        return s
    for en, zh in _TEXT_ZH_ITEMS:
        s = s.replace(en, zh)
    return s


def _style(fig, title=None, height=360):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=12, color="#374151"),
        margin=dict(l=12, r=16, t=44, b=12),
        height=height,
        hovermode="x unified",
        title=dict(text=title, font=dict(size=14.5, color="#1e1b4b"), x=0.01, xanchor="left") if title else None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1e1b4b", font=dict(color="#ffffff", size=12), bordercolor="#1e1b4b"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e5e7eb", tickfont=dict(color="#6b7280"))
    fig.update_yaxes(gridcolor="#f1f5f9", zeroline=False, linecolor="#e5e7eb", tickfont=dict(color="#6b7280"))
    return fig


def _scatter_line(x, y, color=C_ACCENT, mode="lines+markers", name=None):
    return go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(color=color, width=3),
                      marker=dict(size=8, color=color))


# ---------------------------------------------------------------- 通用渲染
def html_report(df, findings, answers, insights, figure_builder, title="数据分析报告",
                subtitle="", correlation_fig=None, plotly_js=None, eda_figs=None):
    """figure_builder(insight_id, df) -> plotly Figure；返回完整 HTML 字符串。

    correlation_fig: 可选，关键关系 section 顶部的大图（如 Spearman 相关条形图）。
    eda_figs: 可选，描述性分析图列表（每个是带标题的 plotly Figure），渲染成 2×2 网格。
    """
    js = plotly_js or get_plotlyjs()
    corr_html = ""
    if correlation_fig is not None:
        corr_html = correlation_fig.to_html(full_html=False, include_plotlyjs=False)

    eda_html = ""
    if eda_figs:
        cells = "".join(
            f'<div class="grid-cell">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
            for fig in eda_figs)
        eda_html = f'<section><h2>数据概览 · 描述性分析</h2><div class="grid2">{cells}</div></section>'

    # ---- 概览 KPI ----
    n_rows, n_cols = df.shape
    kpis = [
        ("数据行数", f"{n_rows:,}"),
        ("字段数", str(n_cols)),
        ("质量问题", str(len(findings))),
        ("自主洞察", str(len(insights))),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-num">{v}</div><div class="kpi-label">{k}</div></div>'
        for k, v in kpis)

    # ---- 数据质量表 ----
    sev_badge = {"P0": '<span class="badge badge-p0">P0</span>',
                 "P1": '<span class="badge badge-p1">P1</span>'}
    rows_f = "".join(
        f'<tr><td>{sev_badge.get(f.get("severity",""), f.get("severity",""))}</td>'
        f'<td><b>{_zh_wrap(f.get("field",""))}</b>'
        f'<br><span class="ev">{f.get("field","")}</span>'
        f'<br><span class="tag">{f.get("issue_type","")}</span></td>'
        f'<td>{f.get("description","")}<br><span class="ev">📌 {f.get("evidence","")}</span></td></tr>'
        for f in findings)

    # ---- 关键关系 ----
    rows_a = "".join(
        f'<div class="rel-item"><div class="rel-q">{_zh_text(a.get("question",""))}</div>'
        f'<div class="rel-a">{_zh_text(a.get("answer",""))}</div>'
        f'<div class="ev">📌 {_zh_text(a.get("evidence",""))}</div></div>'
        for a in answers)

    # ---- 核心业务结论（结论先行） ----
    conclusions = "".join(
        f'<div class="concl"><span class="concl-n">{i+1}</span>'
        f'<div class="concl-body"><div class="concl-text">{_zh_text(ins.get("title",""))}</div>'
        f'<div class="concl-act">💡 {_zh_text(ins.get("action",""))}</div></div></div>'
        for i, ins in enumerate(insights))

    # ---- 洞察卡片 ----
    cards = ""
    for ins in insights:
        fid = ins.get("id", "")
        fig = figure_builder(fid, df)
        fig_div = fig.to_html(full_html=False, include_plotlyjs=False)
        cards += (
            f'<div class="card">'
            f'<div class="card-body">'
            f'<h3>{_zh_text(ins.get("title",""))}</h3>'
            f'<p>{_zh_text(ins.get("insight",""))}</p>'
            f'<div class="ev">📊 {_zh_text(ins.get("evidence",""))}</div>'
            f'<div class="act"><span class="act-label">💡 业务建议</span>{_zh_text(ins.get("action",""))}</div>'
            f'</div>'
            f'<div class="card-chart">{fig_div}</div>'
            f'</div>')

    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>{js}</script>
<style>
*{{box-sizing:border-box}}
body{{font-family:{FONT};margin:0;background:#f8fafc;color:#1f2937;line-height:1.65}}
.wrap{{max-width:980px;margin:0 auto;padding:0 20px 60px}}
header{{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;padding:36px 20px 28px}}
header .wrap{{padding:0 20px}}
h1{{margin:0 0 6px;font-size:26px}}
.sub{{opacity:.85;font-size:14px}}
.kpis{{display:flex;gap:14px;margin:22px 0 6px;flex-wrap:wrap}}
.kpi{{background:#fff;border-radius:12px;padding:16px 22px;flex:1;min-width:130px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi-num{{font-size:26px;font-weight:700;color:#4f46e5}}
.kpi-label{{font-size:13px;color:#6b7280}}
section{{background:#fff;border-radius:14px;padding:24px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
h2{{margin:0 0 16px;font-size:19px;color:#1e3a8a;display:flex;align-items:center;gap:8px}}
h2::before{{content:"";width:4px;height:20px;background:#4f46e5;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:14px;table-layout:fixed}}
th,td{{border-bottom:1px solid #eef2f7;padding:10px 12px;text-align:left;vertical-align:top;word-wrap:break-word;overflow-wrap:anywhere}}
th{{color:#6b7280;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.badge{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700;color:#fff;white-space:nowrap}}
.badge-p0{{background:#ef4444}}.badge-p1{{background:#f59e0b}}
.tag{{display:inline-block;margin-top:4px;padding:1px 8px;border-radius:6px;background:#f1f5f9;color:#475569;font-size:11.5px;font-weight:600}}
.rel-item{{padding:12px 0;border-bottom:1px solid #eef2f7}}
.rel-q{{color:#6b7280;font-size:13px}}
.rel-a{{font-weight:600;margin:4px 0;color:#1f2937}}
.card{{display:flex;gap:20px;border:1px solid #eef2f7;border-radius:12px;padding:18px;margin:16px 0;align-items:center}}
.card-body{{flex:1;min-width:0}}
.card h3{{margin:0 0 8px;font-size:16px;color:#1e3a8a}}
.card p{{margin:0 0 8px;color:#374151}}
.card-chart{{width:46%;min-width:300px}}
.ev{{color:#6b7280;font-size:13px}}
.act{{color:#065f46;font-size:13.5px;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:10px 12px;margin-top:10px}}
.act-label{{display:inline-block;background:#10b981;color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-radius:12px;margin-right:8px;vertical-align:1px}}
.exec{{background:linear-gradient(135deg,#eef2ff,#f5f3ff);border:1px solid #ddd6fe}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:4px}}
.grid-cell{{min-width:0}}
.concl{{display:flex;gap:14px;align-items:flex-start;padding:14px 4px;border-bottom:1px dashed #ddd6fe}}
.concl:last-child{{border-bottom:none}}
.concl-n{{flex-shrink:0;width:30px;height:30px;border-radius:50%;background:#4f46e5;color:#fff;font-weight:700;font-size:15px;display:flex;align-items:center;justify-content:center;margin-top:2px}}
.concl-text{{font-weight:700;font-size:15.5px;color:#1e1b4b}}
.concl-act{{color:#047857;font-size:13.5px;margin-top:4px}}
footer{{text-align:center;color:#9ca3af;font-size:12px;margin-top:30px}}
@media(max-width:760px){{.card{{flex-direction:column}}.card-chart{{width:100%}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="wrap"><h1>📊 {title}</h1><div class="sub">{subtitle}</div>
<div class="kpis">{kpi_html}</div></div></header>
<div class="wrap">
<section class="exec"><h2>核心业务结论</h2>{conclusions}</section>

{eda_html}

<section><h2>一、数据质量</h2>
<table>
<colgroup><col style="width:64px"><col style="width:200px"><col></colgroup>
<thead><tr><th>级别</th><th>字段 / 类型</th><th>说明与证据</th></tr></thead>
<tbody>{rows_f}</tbody></table></section>

<section><h2>二、关键关系</h2>{corr_html}{rows_a}</section>

<section><h2>三、自主洞察</h2>{cards}</section>

<footer>本报告由数据分析 Agent 生成 · 图表为 Plotly 交互式（可悬停/缩放）· 全部数字来自 pandas 真实计算</footer>
</div>
</body></html>"""


if __name__ == "__main__":
    print("report_builder 是模块，请从调用脚本 import html_report。")
