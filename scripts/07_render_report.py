"""L6 — 리포트 HTML 렌더링.

    python scripts/07_render_report.py --strategy value --top 5

리포트는 1~2주마다 다시 만든다. 손으로 HTML을 쓰면 그때마다 옮겨 적으며
수치가 틀어지므로, 데이터에서 직접 렌더링한다.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import report, store, strategies  # noqa: E402

CAT_COLOR = {"성장성": "growth", "수익성": "profit", "안정성": "stable",
             "밸류에이션": "value", "모멘텀·수급": "mom"}

CSS = """
:root{
  --paper:#F1F2F5; --surface:#FFFFFF; --surface-2:#F8F9FB;
  --ink:#171A21; --ink-2:#4A5160; --ink-3:#79808F;
  --rule:#DCDFE6; --rule-strong:#BFC5D0;
  --accent:#294A75; --accent-soft:#E6ECF4;
  --up:#C6353E; --down:#1B58B8;
  --g-ap:#177A52; --g-ap-bg:#E2F2EA; --g-a:#2F8A63; --g-a-bg:#E7F3EC;
  --g-b:#66707F; --g-b-bg:#ECEEF2; --g-c:#A9761B; --g-c-bg:#F8F0DE;
  --g-d:#BE3A2E; --g-d-bg:#FAE7E4;
  --growth:#2E7D6B; --profit:#41689C; --stable:#5E6B82;
  --value:#96652C; --mom:#8A4A67;
  --warn-bg:#FCF4E6; --warn-bd:#E0C289; --warn-ink:#7A5510;
  --sans:'IBM Plex Sans KR','Malgun Gothic',system-ui,sans-serif;
  --serif:'Noto Serif KR',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101219; --surface:#181B22; --surface-2:#1D2129;
  --ink:#E3E6EC; --ink-2:#A5ACBA; --ink-3:#7B8290;
  --rule:#2A2F3A; --rule-strong:#3C4351; --accent:#8FB1E2; --accent-soft:#1D2735;
  --up:#E4666E; --down:#5E93E8;
  --g-ap:#43BA88; --g-ap-bg:#12291F; --g-a:#55A984; --g-a-bg:#152520;
  --g-b:#8C95A5; --g-b-bg:#21252D; --g-c:#D4A552; --g-c-bg:#2A2314;
  --g-d:#E36E61; --g-d-bg:#2C1A17;
  --growth:#4EA891; --profit:#6E9AD6; --stable:#8B98AE;
  --value:#C79251; --mom:#BE7A9B;
  --warn-bg:#282112; --warn-bd:#57471F; --warn-ink:#DEB86A;
}}
:root[data-theme="dark"]{
  --paper:#101219; --surface:#181B22; --surface-2:#1D2129;
  --ink:#E3E6EC; --ink-2:#A5ACBA; --ink-3:#7B8290;
  --rule:#2A2F3A; --rule-strong:#3C4351; --accent:#8FB1E2; --accent-soft:#1D2735;
  --up:#E4666E; --down:#5E93E8;
  --g-ap:#43BA88; --g-ap-bg:#12291F; --g-a:#55A984; --g-a-bg:#152520;
  --g-b:#8C95A5; --g-b-bg:#21252D; --g-c:#D4A552; --g-c-bg:#2A2314;
  --g-d:#E36E61; --g-d-bg:#2C1A17;
  --growth:#4EA891; --profit:#6E9AD6; --stable:#8B98AE;
  --value:#C79251; --mom:#BE7A9B;
  --warn-bg:#282112; --warn-bd:#57471F; --warn-ink:#DEB86A;
}
*{box-sizing:border-box;}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:14.5px;line-height:1.7;-webkit-font-smoothing:antialiased;}
.wrap{max-width:940px;margin:0 auto;padding:0 22px 80px;}
.mast{padding:48px 0 22px;border-bottom:2px solid var(--ink);}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3);display:flex;gap:13px;flex-wrap:wrap;}
h1{font-family:var(--serif);font-weight:700;font-size:clamp(27px,5vw,40px);
  line-height:1.25;margin:14px 0 0;letter-spacing:-.015em;}
.deck{color:var(--ink-2);margin:14px 0 0;max-width:62ch;font-size:15px;}
.deck b{color:var(--ink);font-weight:600;}
.tw{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);margin-top:26px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:600px;}
caption{caption-side:top;text-align:left;font-family:var(--mono);font-size:10.5px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
  padding:11px 14px 8px;border-bottom:1px solid var(--rule);}
th{text-align:left;font-weight:600;font-size:11.5px;letter-spacing:.05em;
  color:var(--ink-2);padding:9px 13px;border-bottom:1px solid var(--rule-strong);
  background:var(--surface-2);white-space:nowrap;}
td{padding:9px 13px;border-bottom:1px solid var(--rule);vertical-align:middle;}
tbody tr:last-child td{border-bottom:0;}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;}
.g{font-family:var(--mono);font-size:11px;font-weight:600;padding:1px 5px;
  border-radius:2px;display:inline-block;line-height:1.6;}
.g.ap{color:var(--g-ap);background:var(--g-ap-bg);}
.g.a{color:var(--g-a);background:var(--g-a-bg);}
.g.b{color:var(--g-b);background:var(--g-b-bg);}
.g.c{color:var(--g-c);background:var(--g-c-bg);}
.g.d{color:var(--g-d);background:var(--g-d-bg);}
.card{background:var(--surface);border:1px solid var(--rule-strong);margin-top:30px;}
.hd{display:flex;align-items:flex-start;gap:18px;flex-wrap:wrap;
  padding:17px 20px 14px;border-bottom:2px solid var(--ink);}
.nm{font-family:var(--serif);font-weight:700;font-size:21px;}
.tk{font-family:var(--mono);font-size:12px;color:var(--ink-3);margin-left:8px;}
.sub{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);margin-top:5px;
  font-variant-numeric:tabular-nums;}
.vd{margin-left:auto;text-align:right;}
.sc{font-family:var(--mono);font-size:30px;font-weight:600;line-height:1;
  font-variant-numeric:tabular-nums;}
.lb{font-family:var(--serif);font-weight:700;font-size:14px;margin-top:4px;}
.lb.ok{color:var(--g-ap);} .lb.no{color:var(--g-d);}
.st{font-family:var(--mono);font-size:10px;color:var(--ink-3);letter-spacing:.07em;margin-top:3px;}
.band{display:flex;flex-wrap:wrap;border-bottom:1px solid var(--rule);}
.cell{flex:1 1 150px;padding:12px 16px;border-right:1px solid var(--rule);}
.cell:last-child{border-right:0;}
.ct{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);display:flex;align-items:center;gap:6px;}
.sw{width:3px;height:11px;display:inline-block;}
.cv{font-family:var(--mono);font-size:20px;font-weight:600;margin-top:6px;
  font-variant-numeric:tabular-nums;}
.sec{padding:15px 20px;border-bottom:1px solid var(--rule);}
.sec:last-child{border-bottom:0;}
h5{font-family:var(--mono);font-size:10px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 10px;font-weight:600;}
.contrib{display:grid;grid-template-columns:auto 1fr auto auto;gap:6px 12px;align-items:center;}
.contrib .lbl{font-size:12.5px;color:var(--ink-2);white-space:nowrap;}
.track{height:9px;background:var(--surface-2);border:1px solid var(--rule);position:relative;}
.fill{position:absolute;inset:0 auto 0 0;}
.contrib .w,.contrib .p{font-family:var(--mono);font-size:12px;
  font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;}
.contrib .p{font-weight:600;min-width:3.4em;}
.contrib .w{color:var(--ink-3);min-width:3.2em;}
.two{display:grid;grid-template-columns:1fr 1fr;}
.two>.sec{border-right:1px solid var(--rule);}
.two>.sec:last-child{border-right:0;}
ul.ev{list-style:none;margin:0;padding:0;display:grid;gap:6px;}
ul.ev li{display:grid;grid-template-columns:auto 1fr;gap:8px;font-size:12.8px;line-height:1.6;}
ul.ev li::before{content:"+";color:var(--g-ap);font-family:var(--mono);font-weight:600;}
ul.ev.con li::before{content:"−";color:var(--g-d);}
ul.ev.brk li::before{content:"!";color:var(--g-c);}
.ladder{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));}
.ladder>div{padding:12px 18px;border-right:1px solid var(--rule);}
.ladder>div:last-child{border-right:0;}
.ladder .lp{font-family:var(--mono);font-size:17px;font-weight:600;
  font-variant-numeric:tabular-nums;margin-top:3px;}
.ladder .lb2{font-size:11.5px;color:var(--ink-3);margin-top:4px;line-height:1.5;}
.note{border:1px solid var(--warn-bd);background:var(--warn-bg);color:var(--warn-ink);
  padding:11px 14px;margin-top:22px;font-size:13.2px;line-height:1.7;border-radius:2px;}
.note b{font-weight:600;}
.note .h{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:600;display:block;margin-bottom:4px;opacity:.85;}
svg{display:block;max-width:100%;height:auto;}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:12.5px;color:var(--ink-3);line-height:1.75;}
@media (max-width:640px){
  .two{grid-template-columns:1fr;} .two>.sec{border-right:0;}
  .cell,.ladder>div{border-right:0;border-bottom:1px solid var(--rule);}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
"""


def gclass(g):
    return {"A+": "ap", "A": "a", "B": "b", "C": "c", "D": "d"}.get(g, "b")


def esc(x):
    return html.escape(str(x))


def band_svg(b: dict) -> str:
    """자기 과거 밸류에이션 밴드에서 현재 위치. 축은 백분위 0~100."""
    if b.get("per_pos") is None:
        return ('<p style="font-size:12.5px;color:var(--ink-3);margin:0;">'
                '밴드 산출 불가 — 시세 또는 순이익 이력 부족</p>')
    pos = b["per_pos"]
    W, H, L, R = 620, 74, 12, 12
    span = W - L - R

    def x(p):
        return L + span * p / 100

    ticks = [(25, b["per_p25"]), (50, b["per_p50"]), (75, b["per_p75"])]
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="2년 PER 밴드 내 현재 위치 '
             f'{pos:.0f}백분위">']
    parts.append(f'<rect x="{L}" y="26" width="{span}" height="12" fill="var(--surface-2)" '
                 f'stroke="var(--rule)"/>')
    parts.append(f'<rect x="{x(0):.1f}" y="26" width="{x(25)-x(0):.1f}" height="12" '
                 f'fill="var(--g-ap-bg)"/>')
    parts.append(f'<rect x="{x(75):.1f}" y="26" width="{x(100)-x(75):.1f}" height="12" '
                 f'fill="var(--g-d-bg)"/>')
    for p, v in ticks:
        parts.append(f'<line x1="{x(p):.1f}" y1="24" x2="{x(p):.1f}" y2="40" '
                     f'stroke="var(--rule-strong)"/>')
        parts.append(f'<text x="{x(p):.1f}" y="53" text-anchor="middle" '
                     f'font-family="IBM Plex Mono, monospace" font-size="10.5" '
                     f'fill="var(--ink-3)">{p}p · {v:.1f}배</text>')
    parts.append(f'<text x="{L}" y="20" font-family="IBM Plex Mono, monospace" '
                 f'font-size="10" fill="var(--g-ap)">싼 구간</text>')
    parts.append(f'<text x="{L+span}" y="20" text-anchor="end" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10" '
                 f'fill="var(--g-d)">비싼 구간</text>')
    mx = x(pos)
    parts.append(f'<polygon points="{mx:.1f},22 {mx-5:.1f},14 {mx+5:.1f},14" '
                 f'fill="var(--accent)"/>')
    parts.append(f'<line x1="{mx:.1f}" y1="22" x2="{mx:.1f}" y2="42" '
                 f'stroke="var(--accent)" stroke-width="2"/>')
    anchor = "start" if pos < 20 else ("end" if pos > 80 else "middle")
    parts.append(f'<text x="{mx:.1f}" y="68" text-anchor="{anchor}" '
                 f'font-family="IBM Plex Mono, monospace" font-size="11" font-weight="600" '
                 f'fill="var(--accent)">현재 {pos:.0f}p</text>')
    parts.append("</svg>")
    return "".join(parts)


def card(r: dict) -> str:
    cats = [c for c in r["categories"] if c["score"] is not None]
    band = "".join(
        f'<div class="cell"><div class="ct">'
        f'<span class="sw" style="background:var(--{CAT_COLOR[c["name"]]});"></span>'
        f'{esc(c["name"])} <span class="g {gclass(c["grade"])}">{esc(c["grade"])}</span></div>'
        f'<div class="cv">{c["score"]:.0f}</div></div>' for c in cats)

    mx = max((c["points"] or 0) for c in r["contrib"]) or 1
    contrib = "".join(
        f'<span class="lbl">{esc(c["cat"])}</span>'
        f'<span class="track"><span class="fill" style="width:{(c["points"] or 0)/mx*100:.0f}%;'
        f'background:var(--{CAT_COLOR[c["cat"]]});"></span></span>'
        f'<span class="w">{c["score"]:.0f} × {c["weight"]}%</span>'
        f'<span class="p">{c["points"]:.1f}</span>'
        for c in r["contrib"] if c["points"] is not None)

    ladder = "".join(
        f'<div><h5>{esc(e["step"])} · {e["weight"]*100:.0f}%</h5>'
        f'<div class="lp">{e["price"]:,.0f}원</div>'
        f'<div class="lb2">{esc(e["basis"])}</div></div>' for e in r["entry"])

    pros = "".join(f"<li><span>{esc(x)}</span></li>" for x in r["pros"])
    cons = "".join(f"<li><span>{esc(x)}</span></li>" for x in r["cons"])
    brk = "".join(f"<li><span>{esc(x)}</span></li>" for x in r["breakers"])

    warn = ""
    if r["coverage"] is not None and r["coverage"] < 4:
        warn = (f'<div class="sec"><div class="note" style="margin:0;">'
                f'<span class="h">점수 신뢰도</span>부문 커버리지 '
                f'<b>{r["coverage"]}/4</b> — 결측 지표가 있어 남은 부문끼리 '
                f'가중치를 재정규화했습니다.</div></div>')

    return f"""
<article class="card">
  <div class="hd">
    <div>
      <span class="nm">{esc(r['name'])}</span><span class="tk">{esc(r['ticker'])} · {esc(r['market'])} · {esc(r['track'])}</span>
      <div class="sub">{r['price']:,.0f}원 · 시총 {r['market_cap']/1e12:.1f}조 ·
        재무 {esc(r['fin_period'])} (접수 {esc(r['rcept_dt'])})</div>
    </div>
    <div class="vd">
      <div class="sc">{r['score']:.1f}</div>
      <div class="lb {'ok' if r['gate_pass'] else 'no'}">{'게이트 통과' if r['gate_pass'] else '게이트 탈락'}</div>
      <div class="st">기본 {r['base_score']:.1f} · 감점 {r['penalty']:.0f}</div>
    </div>
  </div>
  <div class="band">{band}</div>
  <div class="sec"><h5>점수 기여도 · {esc(r['strategy_kr'])} 가중치</h5>
    <div class="contrib">{contrib}</div></div>
  <div class="sec"><h5>자기 과거 밸류에이션 밴드 · 최근 2년 PER</h5>{band_svg(r['band'])}</div>
  <div class="ladder">{ladder}</div>
  <div class="two">
    <div class="sec"><h5>찬성 근거</h5><ul class="ev">{pros}</ul></div>
    <div class="sec"><h5>반대 근거 · 같은 개수로 강제</h5><ul class="ev con">{cons}</ul></div>
  </div>
  <div class="sec"><h5>논거 훼손 조건 · 발생 시 가격과 무관하게 청산</h5>
    <ul class="ev brk">{brk}</ul></div>
  {warn}
</article>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="value")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--out", default="report_value.html")
    args = ap.parse_args()

    g = store.load("grades_latest")
    sc = store.load("strategy_scores")
    px = store.load("prices_daily")
    ttm = store.load("fin_ttm")
    sel = sc[(sc.strategy == args.strategy) & sc.gate_pass].nlargest(args.top, "score")
    total = int((sc.strategy == args.strategy).sum())
    passed = int(((sc.strategy == args.strategy) & sc.gate_pass).sum())
    reports = [report.build(t, g, sc, px, ttm, args.strategy, args.asof)
               for t in sel.ticker]

    label = strategies.LABEL[args.strategy]
    w = " · ".join(f"{report.CAT_KR[c]} {v}"
                   for c, v in strategies.WEIGHTS[args.strategy].items() if v)

    rows = "".join(
        f"<tr><td><b>{esc(r['name'])}</b> "
        f"<span style='font-family:var(--mono);font-size:11.5px;color:var(--ink-3);'>"
        f"{esc(r['ticker'])}</span></td>"
        f"<td class='num'>{r['score']:.1f}</td>"
        + "".join(
            f"<td class='num'>{c['score']:.0f} "
            f"<span class='g {gclass(c['grade'])}'>{esc(c['grade'])}</span></td>"
            if c["score"] is not None else "<td class='num'>—</td>"
            for c in r["categories"])
        + f"<td class='num'>{r['band']['per_pos']:.0f}p</td>"
          if r["band"]["per_pos"] is not None else "<td class='num'>—</td>"
        for r in reports)
    # 위 표현식이 조건 우선순위 때문에 깨지므로 명시적으로 다시 만든다
    rows = ""
    for r in reports:
        cells = "".join(
            (f"<td class='num'>{c['score']:.0f} "
             f"<span class='g {gclass(c['grade'])}'>{esc(c['grade'])}</span></td>")
            if c["score"] is not None else "<td class='num'>—</td>"
            for c in r["categories"])
        pos = (f"{r['band']['per_pos']:.0f}p"
               if r["band"]["per_pos"] is not None else "—")
        rows += (f"<tr><td><b>{esc(r['name'])}</b> <span style=\"font-family:var(--mono);"
                 f"font-size:11.5px;color:var(--ink-3);\">{esc(r['ticker'])}</span></td>"
                 f"<td class='num'>{r['score']:.1f}</td>{cells}"
                 f"<td class='num'>{pos}</td></tr>")

    cards = "".join(card(r) for r in reports)

    doc = f"""<title>{label} 스크리닝 리포트</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="mast">
  <div class="eyebrow"><span>{esc(args.asof)} 기준</span><span>·</span>
    <span>{esc(label)}</span><span>·</span><span>KOSPI·KOSDAQ 시총 상위 200</span></div>
  <h1>{esc(label)} 상위 {len(reports)}종목</h1>
  <p class="deck">유니버스 <b>{total}종목</b> 중 게이트를 통과한 <b>{passed}종목</b>에서
  점수 상위 {len(reports)}개. 재무는 각 종목의 <b>공시 접수일 기준</b>으로 그 시점에
  알 수 있었던 값만 썼습니다. 가중치는 {esc(w)}.</p>
</header>

<div class="tw"><table>
  <caption>한눈에 보기 — 부문 점수는 동종업종 내 백분위</caption>
  <thead><tr><th>종목</th><th class="num">점수</th>
    <th class="num">성장성</th><th class="num">수익성</th><th class="num">안정성</th>
    <th class="num">밸류</th><th class="num">모멘텀</th><th class="num">밴드위치</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>

<div class="note">
  <span class="h">지금 단계에서 비어 있는 것</span>
  이 리포트는 <b>정량 계층만</b>으로 만들어졌습니다. 설계상 함께 들어가야 할
  <b>증권사 목표주가·투자의견</b>(한경컨센서스 수동 취합)과
  <b>공시·IR 기반 정성 근거</b>(사업 로드맵, 경쟁우위, 회사 고유 점검 항목)는 아직 없습니다.
  또한 가중치는 검증 전의 논리값이라, <b>이 점수는 정렬 기준이지 예측값이 아닙니다.</b>
</div>

{cards}

<footer>
  <p>종목 선별 시스템의 산출물이며 특정 종목에 대한 투자 권유가 아닙니다.
  점수는 정해진 규칙의 계산 결과이고 최종 판단과 책임은 사용자에게 있습니다.
  밸류에이션 밴드는 수집한 시세 창(2년)에서 산출했으며 설계 초안의 5년보다 짧습니다.
  진입 가격은 밴드에서 역산한 값으로 매수 지시가 아닙니다.</p>
</footer>
</div>"""

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"저장: {out}  ({len(doc):,} bytes, {len(reports)}종목)")
    for r in reports:
        print(f"  {r['name']:<12} {r['score']:>5.1f}  밴드 {r['band']['per_pos'] or 0:>5.1f}p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
