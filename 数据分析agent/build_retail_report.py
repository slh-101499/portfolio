# -*- coding: utf-8 -*-
"""
零售数据集的报告生成入口：加载 CSV + 分析 JSON，生成 Plotly 交互式自包含报告。
通用逻辑在 report_builder.html_report；本文件只负责「算图 + 装配」。
"""
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from report_builder import html_report, _style, FIELD_ZH

# ---- 统一配色（现代、柔和） ----
INDIGO  = "#6366f1"
VIOLET  = "#8b5cf6"
AMBER   = "#f59e0b"
EMERALD = "#10b981"
ROSE    = "#f43f5e"
CYAN    = "#06b6d4"

_FILL = "rgba(99,102,241,0.14)"      # indigo 渐变填充
_FILL_AMBER = "rgba(245,158,11,0.16)"
_FILL_GREEN = "rgba(16,185,129,0.14)"
_FILL_ROSE  = "rgba(244,63,94,0.12)"


def _correlation_fig(df):
    feats = ['age', 'annual_income', 'basket_size', 'avg_item_price', 'visit_frequency_monthly',
             'loyalty_points', 'online_purchase_pct', 'return_rate', 'discount_sensitivity',
             'satisfaction_score', 'days_since_last_visit', 'customer_segment', 'customer_lifetime_value']
    sp = df[feats + ['transaction_amount']].corr(method='spearman')['transaction_amount'].drop('transaction_amount').sort_values()
    colors = [ROSE if v < 0 else INDIGO for v in sp.values]
    fig = go.Figure(go.Bar(
        x=sp.values, y=[FIELD_ZH.get(f, f) for f in sp.index], orientation='h',
        marker_color=colors, marker_line=dict(color="rgba(0,0,0,0.05)", width=0.5),
        text=[f"{v:+.2f}" for v in sp.values], textposition='outside',
        textfont=dict(size=11, color="#374151"),
        hovertemplate='%{y} → %{x:+.3f}<extra></extra>'))
    fig.add_vline(x=0, line=dict(color="#d1d5db", width=1))
    fig.update_xaxes(title="Spearman ρ", range=[-0.25, 0.95])
    return _style(fig, "各字段与交易金额的 Spearman 秩相关（重尾数据的正确姿势）", height=430)


def _fig_i1(df):
    vf = df.groupby(pd.cut(df['visit_frequency_monthly'], [0, 1, 3, 7, 13, 50, 200],
             labels=['0', '1-2', '3-6', '7-12', '13-49', '50+']), observed=True)['transaction_amount'].mean()
    fig = go.Figure(go.Scatter(x=list(vf.index), y=vf.values, mode='lines+markers',
        line=dict(color=AMBER, width=3.5, shape='spline', smoothing=0.6),
        marker=dict(size=9, color=AMBER, line=dict(color="#fff", width=1.5)),
        fill='tozeroy', fillcolor=_FILL_AMBER,
        hovertemplate='到店 %{x} 次/月：客单价 %{y:.1f} 元<extra></extra>'))
    fig.add_annotation(x='7-12', y=vf['7-12'], text='最佳区间', showarrow=True, arrowhead=2,
                       ax=0, ay=-32, font=dict(size=12, color="#b45309"), bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x='50+', y=vf['50+'], text='回落', showarrow=True, arrowhead=2,
                       ax=0, ay=30, font=dict(size=12, color="#dc2626"), bgcolor="rgba(255,255,255,0.85)")
    fig.update_yaxes(range=[90, 220])
    return _style(fig, "到店频次 → 客单价（非单调）")


def _fig_i2(df):
    db = df.groupby(pd.cut(df['discount_sensitivity'], [0, .3, .5, .7, 1.01],
             labels=['低', '中', '高', '极高']), observed=True).agg(
        amt=('transaction_amount', 'mean'), sat=('satisfaction_score', 'mean'))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(db.index), y=db['amt'], name='客单价',
        line=dict(color=INDIGO, width=3.5, shape='spline', smoothing=0.6),
        mode='lines+markers', marker=dict(size=8), fill='tozeroy', fillcolor=_FILL), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(db.index), y=db['sat'], name='满意度',
        line=dict(color=EMERALD, width=3.5, shape='spline', smoothing=0.6),
        mode='lines+markers', marker=dict(size=8), fill='tozeroy', fillcolor=_FILL_GREEN), secondary_y=True)
    fig.update_yaxes(title_text="客单价(元)", secondary_y=False, title_font_color=INDIGO, tickfont_color=INDIGO)
    fig.update_yaxes(title_text="满意度", secondary_y=True, title_font_color=EMERALD, tickfont_color=EMERALD)
    return _style(fig, "折扣敏感度：花得少，却更满意")


def _fig_i3(df):
    seg = df.groupby('customer_segment').agg(n=('transaction_id', 'count'), rev=('transaction_amount', 'sum'))
    n_pct = seg['n'] / seg['n'].sum() * 100
    r_pct = seg['rev'] / seg['rev'].sum() * 100
    labels = [f'客群{i}' for i in seg.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=n_pct, name='记录占比 %', marker_color=INDIGO,
        marker_line=dict(color="rgba(0,0,0,0.05)", width=0.5),
        text=[f"{v:.1f}%" for v in n_pct], textposition='outside', textfont=dict(size=11)))
    fig.add_trace(go.Bar(x=labels, y=r_pct, name='营收占比 %', marker_color=AMBER,
        marker_line=dict(color="rgba(0,0,0,0.05)", width=0.5),
        text=[f"{v:.1f}%" for v in r_pct], textposition='outside', textfont=dict(size=11)))
    fig.update_layout(barmode='group', bargap=0.22, bargroupgap=0.06)
    return _style(fig, "Pareto：segment 3 占 33% 记录、贡献 45.7% 营收")


def _fig_i4(df):
    sb = df.groupby(pd.cut(df['satisfaction_score'], [0, 3, 5, 7, 8, 10.01],
             labels=['1-3', '3-5', '5-7', '7-8', '8-10']), observed=True).agg(
        ret=('return_rate', 'mean'), ltv=('customer_lifetime_value', 'mean'))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(sb.index), y=sb['ret'], name='退货率',
        line=dict(color=ROSE, width=3.5, shape='spline', smoothing=0.6),
        mode='lines+markers', marker=dict(size=8), fill='tozeroy', fillcolor=_FILL_ROSE), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(sb.index), y=sb['ltv'], name='LTV',
        line=dict(color=EMERALD, width=3.5, shape='spline', smoothing=0.6),
        mode='lines+markers', marker=dict(size=8), fill='tozeroy', fillcolor=_FILL_GREEN), secondary_y=True)
    fig.update_yaxes(title_text="退货率", secondary_y=False, title_font_color=ROSE, tickfont_color=ROSE)
    fig.update_yaxes(title_text="LTV", secondary_y=True, title_font_color=EMERALD, tickfont_color=EMERALD)
    return _style(fig, "满意度是领先指标：越低退货越高、LTV 越低")


def _fig_i5(df):
    piv = df.pivot_table(index='store_type', columns='loyalty_member',
                         values='transaction_amount', aggfunc='mean')
    prem = (piv['Yes'] - piv['No']).sort_values()
    store_zh = {"Discount": "折扣店", "Luxury": "奢侈店", "Premium": "高端店", "Standard": "标准店"}
    y_labels = [store_zh.get(s, s) for s in prem.index]
    colors = [VIOLET if i == len(prem) - 1 else INDIGO for i in range(len(prem))]
    fig = go.Figure(go.Bar(x=prem.values, y=y_labels, orientation='h',
        marker_color=colors, marker_line=dict(color="rgba(0,0,0,0.05)", width=0.5),
        text=[f"+{v:.1f}" for v in prem.values], textposition='outside',
        textfont=dict(size=11, color="#374151"),
        hovertemplate='%{y} 会员溢价 %{x:.1f} 元<extra></extra>'))
    fig.update_xaxes(title="会员溢价(元)")
    return _style(fig, "会员溢价随门店档次放大")


FIG_BUILDERS = {"I1": _fig_i1, "I2": _fig_i2, "I3": _fig_i3, "I4": _fig_i4, "I5": _fig_i5}


# ---- 描述性分析（原始数据概览，2×2 网格） ----
def _eda_monthly(df):
    """月度客单价均值 vs 中位数：两条线始终分离，直观暴露重尾（极端值抬高均值）。"""
    d = pd.to_datetime(df['transaction_date'], errors='coerce')
    m = df.assign(ym=d.dt.to_period('M')).groupby('ym').agg(
        mean=('transaction_amount', 'mean'), med=('transaction_amount', 'median'))
    x = m.index.to_timestamp()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=m['mean'], name='客单价均值',
        line=dict(color=INDIGO, width=3), mode='lines+markers', marker=dict(size=4),
        hovertemplate='%{x|%Y-%m} 均值 %{y:.1f} 元<extra></extra>'))
    fig.add_trace(go.Scatter(x=x, y=m['med'], name='客单价中位数',
        line=dict(color=AMBER, width=3, dash='dash'), mode='lines',
        fill='tonexty', fillcolor=_FILL_AMBER,
        hovertemplate='%{x|%Y-%m} 中位数 %{y:.1f} 元<extra></extra>'))
    fig.update_yaxes(title="客单价(元)")
    fig.update_xaxes(tickformat='%Y-%m')
    return _style(fig, "月度客单价走势：均值长期高于中位数（重尾信号）", height=340)


def _eda_amount_hist(df):
    """交易金额分布：右偏重尾，中位数 61 元 vs 均值 114 元，极端值高达 7.9 万。
    先 numpy 分箱再用 Bar 渲染，避免把 80 万条原始值序列化进 HTML。"""
    a = df['transaction_amount']
    p99 = float(a.quantile(0.99))
    clipped = a[a <= p99]
    counts, edges = np.histogram(clipped, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]),
        marker_color=INDIGO, marker_line=dict(color="#fff", width=0.5),
        hovertemplate='客单价 %{x:.0f} 元：%{y:,} 笔<extra></extra>'))
    fig.add_vline(x=a.median(), line=dict(color=AMBER, width=2, dash='dash'))
    fig.add_vline(x=a.mean(), line=dict(color=ROSE, width=2, dash='dash'))
    fig.add_annotation(x=a.median(), y=1.0, yref='paper', text=f'中位数 {a.median():.0f}',
                       showarrow=False, yshift=8, font=dict(color='#b45309', size=11))
    fig.add_annotation(x=a.mean(), y=1.0, yref='paper', text=f'均值 {a.mean():.0f}',
                       showarrow=False, yshift=22, font=dict(color='#dc2626', size=11))
    fig.update_xaxes(title="交易金额(元)")
    fig.update_yaxes(title="笔数")
    return _style(fig, f"交易金额重尾分布（截断 99 分位 {p99:.0f} 元，极端最高 {a.max():.0f} 元）", height=340)


def _eda_store(df):
    """门店类型：交易量 vs 客单价中位数（奢侈店量少价高）。"""
    order = ['Discount', 'Standard', 'Premium', 'Luxury']
    store_zh = {"Discount": "折扣店", "Standard": "标准店", "Premium": "高端店", "Luxury": "奢侈店"}
    g = df.groupby('store_type').agg(n=('transaction_id', 'count'),
                                     med=('transaction_amount', 'median')).reindex(order)
    labels = [store_zh[s] for s in g.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=g['n'], name='交易量(笔)', marker_color=INDIGO,
        marker_line=dict(color="rgba(0,0,0,0.05)", width=0.5),
        text=[f"{v/1000:.0f}k" for v in g['n']], textposition='outside', textfont=dict(size=11)),
        secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=g['med'], name='客单价中位数(元)',
        line=dict(color=AMBER, width=3), mode='lines+markers', marker=dict(size=8)),
        secondary_y=True)
    fig.update_yaxes(title_text="交易量(笔)", secondary_y=False)
    fig.update_yaxes(title_text="客单价中位数(元)", secondary_y=True)
    fig = _style(fig, "门店类型：交易量 vs 客单价中位数", height=340)
    fig.update_yaxes(title_font_color=INDIGO, tickfont_color=INDIGO, secondary_y=False)
    fig.update_yaxes(title_font_color=AMBER, tickfont_color=AMBER, secondary_y=True)
    return fig


def _eda_payment(df):
    """支付方式分布（环形饼图）。"""
    order = ['Cash', 'Debit', 'Mobile', 'Credit']
    pay_zh = {"Cash": "现金", "Debit": "借记卡", "Mobile": "移动支付", "Credit": "信用卡"}
    g = df['payment_method'].value_counts().reindex(order)
    labels = [pay_zh[s] for s in g.index]
    fig = go.Figure(go.Pie(labels=labels, values=g.values, hole=0.45,
        marker=dict(colors=[INDIGO, VIOLET, EMERALD, AMBER], line=dict(color="#fff", width=2)),
        textinfo='label+percent', textfont=dict(size=12),
        hovertemplate='%{label}：%{value:,} 笔（%{percent}）<extra></extra>'))
    fig.add_annotation(text=f'{g.sum():,} 笔', x=0.5, y=0.5, showarrow=False,
                       font=dict(size=15, color="#1e1b4b"))
    fig = _style(fig, "支付方式分布：现金仍是主力", height=340)
    fig.update_layout(showlegend=False)
    return fig


EDA_FIGS = [_eda_monthly, _eda_store, _eda_payment, _eda_amount_hist]


def build(csv_path, json_path, out_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    for c in ['gender', 'store_type', 'product_category', 'loyalty_member',
              'payment_method', 'referral_source']:
        if c in df.columns:
            df[c] = df[c].str.strip()
    data = json.load(open(json_path, encoding='utf-8'))
    html = html_report(
        df,
        findings=data['findings'],
        answers=data['answers'],
        insights=data['insights'],
        figure_builder=lambda fid, d: FIG_BUILDERS[fid](d),
        title="合成零售数据 · 深度分析报告",
        subtitle="800,000 行 × 22 列 · 目标指标 transaction_amount · 图表可交互",
        correlation_fig=_correlation_fig(df),
        eda_figs=[f(df) for f in EDA_FIGS],
    )
    open(out_path, 'w', encoding='utf-8').write(html)
    print(f"已生成 {out_path}（{len(html.encode())/1024/1024:.1f} MB）")


if __name__ == "__main__":
    build("synthetic_retail_20250901.csv",
          "synthetic_retail_20250901_analysis.json",
          "synthetic_retail_20250901_report_v2.html")
