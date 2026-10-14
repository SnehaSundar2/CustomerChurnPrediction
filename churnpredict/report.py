"""Model reports: a text summary for the terminal and a self-contained HTML report with charts.

Chart conventions: one validated colour per chart, thin bars with a rounded data end, solid hairline
gridlines, text in text colours (never the series colour), a tooltip on every mark, a "show as table"
view for every chart, and light/dark themes.
"""

import html
import math
from datetime import datetime
from pathlib import Path

from . import schema as S
from .exceptions import ModelError

WIDTH = 78
money = S.format_money


# ====================================================================== terminal
def bar(value, top, width=24):
    return "█" * max(1 if value > 0 else 0, round(width * value / top)) if top else ""


def _policy(meta, name):
    return next(p for p in meta["policies"] if p["policy"].startswith(name))


def _verdict(meta):
    test, train, cal = meta["metrics"]["test"], meta["metrics"]["train"], meta["calibration"]
    eco, base = meta["economics"], meta["metrics"]["baseline_cv"]
    model, everyone, perfect = _policy(meta, "Use the model"), _policy(meta, "Contact everybody"), _policy(meta, "Perfect")
    lines = [f"Pick one customer who left and one who stayed at random: the model scores the leaver higher "
             f"{test['roc_auc'] * 100:.0f}% of the time (a coin flip would be 50%)."]
    if meta.get("ceiling"):
        lines[0] += (f" The best any model could do on this synthetic data is {meta['ceiling']['roc_auc'] * 100:.0f}%, "
                     "because who actually leaves involves luck.")
    lines.append(f"At the contact threshold ({test['threshold'] * 100:.1f}% churn chance) the model flags {test['flagged']} of "
                 f"{test['n']} customers. They include {test['tp']} of the {test['tp'] + test['fn']} who left "
                 f"({test['recall']:.0f}%), and {test['precision']:.0f}% of the flagged customers really did leave.")
    if eco["profitable"]:
        extra = (model["profit"] / everyone["profit"] - 1) * 100 if everyone["profit"] > 0 else None
        lines.append(f"Making the retention offer to the model's list earns an expected {money(model['profit'])} on these "
                     f"test customers, against {money(everyone['profit'])} for offering it to everybody"
                     + (f" ({extra:.0f}% more)" if extra is not None else "")
                     + f"; a perfect crystal ball would earn {money(perfect['profit'])}.")
    else:
        lines.append("With these campaign costs no offer pays for itself on average, so the threshold is set above 100%: "
                     "nobody is contacted.")
    lines.append(f"Calling everybody 'will stay' is {test['majority_accuracy']:.0f}% accurate but reaches no one at risk, so "
                 "accuracy alone is not a fair score here.")
    lines.append(f"The probabilities are honest: Brier score {test['brier']:.3f} against {base['brier']:.3f} for always guessing "
                 f"the average churn rate (lower is better)"
                 + (f"; they were calibrated with the {cal['method']} method." if cal.get("used") else
                    "; the model's own probabilities were already well calibrated."))
    gap = train["roc_auc"] - test["roc_auc"]
    if gap > 0.06:
        lines.append(f"Overfitting warning: ROC AUC is {train['roc_auc']:.3f} on training data but {test['roc_auc']:.3f} on test data.")
    else:
        lines.append(f"No serious overfitting: ROC AUC is {train['roc_auc']:.3f} on training data and {test['roc_auc']:.3f} on test data.")
    default = test["default_threshold"]
    lines.append(f"The usual 0.5 cut-off would flag only {default['flagged']} customers and miss "
                 f"{100 - default['recall']:.0f}% of those who leave; the campaign threshold is chosen from the economics instead.")
    return lines


def model_text(meta):
    test = meta["metrics"]["test"]
    rows, eco = meta["rows"], meta["economics"]
    out = ["=" * WIDTH, "  MODEL REPORT",
           f"  {meta['model']}{' (' + meta['model_note'] + ')' if meta.get('model_note') else ''} · trained "
           f"{meta['trained_at'].replace('T', ' ')} in {meta['training_seconds']} s · data: {meta['source']}",
           f"  {rows['raw']} rows read → {rows['clean']} after cleaning → {rows['train']} to train, {rows['test']} to test "
           f"({test['churn_rate']:.1f}% churn)",
           "=" * WIDTH, "", "  ACCURACY ON THE UNSEEN TEST SET",
           f"   ROC AUC       {test['roc_auc']:.3f}       (1.0 = perfect ranking, 0.5 = coin flip)",
           f"   PR AUC        {test['average_precision']:.3f}       (always-guess-average scores {test['churn_rate'] / 100:.3f})",
           f"   Brier score   {test['brier']:.3f}       (honesty of the probabilities, lower is better)",
           f"   At the contact threshold of {test['threshold'] * 100:.1f}%:",
           f"     Precision {test['precision']:.1f}%   (flagged customers who really left)",
           f"     Recall    {test['recall']:.1f}%   (customers who left that were flagged)",
           f"     Flagged   {test['flagged']} of {test['n']} customers", "",
           "  CONFUSION MATRIX (test customers)",
           f"                      flagged   not flagged",
           f"   really left        {test['tp']:>7}   {test['fn']:>11}   (missed)",
           f"   really stayed      {test['fp']:>7}   {test['tn']:>11}   (offer wasted)", ""]
    out += ["  WHAT A RETENTION CAMPAIGN EARNS (test customers)",
            f"   Offer cost {money(eco['cost'])} · customer value {money(eco['value'])} · {eco['save_rate'] * 100:.0f}% of offers work "
            f"→ an offer pays off above a {eco['break_even'] * 100:.1f}% churn chance",
            f"   {'POLICY':<32} {'CONTACTED':>9} {'REACHED':>8} {'PROFIT':>12}"]
    for p in meta["policies"]:
        out.append(f"   {p['policy']:<32} {p['contacted']:>9} {p['churners_reached']:>8} {money(p['profit']):>12}")
    out += ["", "  → " + "\n  → ".join(_verdict(meta)), "", "  LIFT BY RISK DECILE (test customers, most at-risk first)"]
    top = max(r["churn_rate"] for r in meta["lift"]) or 1
    for r in meta["lift"]:
        out.append(f"   {r['decile']:>2}  churn {r['churn_rate']:5.1f}%  lift {r['lift']:.1f}×  caught so far {r['captured']:5.1f}%  "
                   f"{bar(r['churn_rate'], top, 16)}")
    out += ["", "  WHAT THE MODEL RELIES ON (drop in ROC AUC when a feature is scrambled)"]
    top = meta["importance"][0]["importance"] or 1
    for item in meta["importance"][:8]:
        out.append(f"   {item['label']:<26} {item['importance']:6.3f}  {bar(item['importance'], top)}")
    out += ["", "  CHURN BY CONTRACT (test set: real vs predicted)"]
    for s in meta["segments"]["contract"]:
        out.append(f"   {s['group']:<16} real {s['actual']:5.1f}%  predicted {s['predicted']:5.1f}%  ({s['n']} customers)")
    return "\n".join(out)


def comparison_text(meta):
    rows = ["  MODEL COMPARISON (5-fold cross-validation on training data; higher AUC is better)",
            f"  {'MODEL':<28} {'ROC AUC':>8} {'PR AUC':>7} {'BRIER':>7}", f"  {'-' * 28} {'-' * 8} {'-' * 7} {'-' * 7}"]
    for c in meta["comparison"]:
        mark = "  ← chosen" if c["model"] == meta["metrics"]["cv"]["model"] else ""
        rows.append(f"  {c['model']:<28} {c['roc_auc']:>8.3f} {c['average_precision']:>7.3f} {c['brier']:>7.3f}{mark}")
    return "\n".join(rows)


# ====================================================================== HTML
CSS = """
.rep{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--series:#2a78d6;--border:rgba(11,11,11,.10);--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}}
:root[data-theme="dark"] .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}
*{box-sizing:border-box}html,body{margin:0}
body.rep{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px}section{margin-top:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:12px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font-size:13px;font-family:inherit}
svg.chart .label{fill:var(--ink2)}svg.chart .value{fill:var(--ink);font-weight:600}
svg.chart .gridline{stroke:var(--grid)}svg.chart .baseline{stroke:var(--axis)}svg.chart .ref{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:5 4}
svg.chart .line{fill:none;stroke:var(--series);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.mark{fill:var(--series);outline:none}.mark.other{fill:var(--other)}.mark:hover,.mark:focus-visible{fill-opacity:.75}
.mark:focus-visible{stroke:var(--ink);stroke-width:1.5}circle.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}
ul.notes{margin:8px 0 0;padding-left:20px}ul.notes li{margin:4px 0}
details.tv{margin-top:8px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}#tip b{display:block;font-size:15px}
footer{margin-top:32px;color:var(--muted);font-size:12px}@media (max-width:520px){.tile.hero .value{font-size:40px}.grid{grid-template-columns:1fr}}
"""
TOOLTIP_JS = """(function(){var tip=document.getElementById('tip');
function show(el,x,y){tip.textContent='';var b=document.createElement('b');b.textContent=el.getAttribute('data-value');
var s=document.createElement('span');s.textContent=el.getAttribute('data-label');tip.appendChild(b);tip.appendChild(s);
tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,window.innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-value]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.right,r.top);});el.addEventListener('blur',hide);});})();"""


def _e(value):
    return html.escape(str(value), quote=True)


def _tick_ceiling(value, intervals=4):
    """Smallest axis maximum with a 'nice' step (1, 2, 2.5, 5, 10 × a power of ten) that covers `value`."""
    raw = max(value, 1e-9) / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude * intervals
    return 10 * magnitude * intervals


def _nice_axis(lo, hi, intervals=5):
    """(low, high, step) with round numbers that cover [lo, hi]."""
    span = (hi - lo) or 1.0
    raw = span / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if raw <= m * magnitude)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step, step


def _end_rounded(x, y, w, h, r=4):
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def _top_rounded(x, y, w, h, r=4):
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z")


def hbar_chart(points, aria, highlight=None, fmt="{:g}"):
    """points [(label, value)]; with `highlight`, that bar is the series colour and the rest recede."""
    w, row, label_w, right = 420, 30, 190, 58
    h = row * len(points) + 8
    vmax = max(v for _, v in points) or 1
    pw = w - label_w - right
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">',
             f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="0" y2="{h - 4}"/>']
    for i, (label, value) in enumerate(points):
        y = 4 + i * row
        bw = max(pw * value / vmax, 2)
        cls = "mark" + ("" if highlight in (None, label) else " other")
        shown = fmt.format(value)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 19}" text-anchor="end">{_e(label[:28])}</text>'
                     f'<path class="{cls}" tabindex="0" d="{_end_rounded(label_w, y + 6, bw, 18)}" data-value="{_e(shown)}" '
                     f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                     f'<text class="value" x="{label_w + bw + 6:.1f}" y="{y + 19}">{_e(shown)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def column_chart(points, aria, x_title="", unit="customers", fmt="{:g}"):
    """points [(label, value)] as thin columns with round ticks."""
    w, h, left, right, top, bottom = 420, 236, 40, 6, 20, 52
    pw, ph = w - left - right, h - top - bottom
    vmax = max(5, _tick_ceiling(max((v for _, v in points), default=0), 5))
    band = pw / max(len(points), 1)
    bar_w = min(24, band - 2)
    tallest = max(range(len(points)), key=lambda i: points[i][1]) if points else -1
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        y = top + ph - ph * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{vmax * k / 5:g}</text>')
    for i, (label, value) in enumerate(points):
        x = left + band * i + (band - bar_w) / 2
        bh = ph * value / vmax if vmax else 0
        shown = fmt.format(value)
        if value:
            parts.append(f'<path class="mark" tabindex="0" d="{_top_rounded(x, top + ph - bh, bar_w, bh)}" data-value="{_e(shown)} {unit}" '
                         f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>')
        parts.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">{_e(label)}</text>')
        if i == tallest and value:
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{top + ph - bh - 5:.1f}" text-anchor="middle">{_e(shown)}</text>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text></svg>')
    return "".join(parts)


def xy_chart(points, aria, x_title, y_title, x_range=(0, 1), y_range=None, x_fmt="{:.0%}", y_fmt="{:.0%}",
             marker=None, marker_label="", diagonal=False, dots=False):
    """A line (or dots) chart. `points` are (x, y) or (x, y, tooltip_label); `marker` = (x, y) is highlighted."""
    w, h, left, bottom, top, right = 420, 340, 74, 40, 10, 14
    pw, ph = w - left - right, h - top - bottom
    ys = [p[1] for p in points] + ([marker[1]] if marker else [])
    y_lo, y_hi, y_step = (y_range + (None,))[:3] if y_range else _nice_axis(min(0, min(ys)), max(ys))
    if y_step is None:
        y_step = (y_hi - y_lo) / 5
    x_lo, x_hi = x_range
    sx = lambda v: left + pw * (v - x_lo) / ((x_hi - x_lo) or 1)
    sy = lambda v: top + ph - ph * (v - y_lo) / ((y_hi - y_lo) or 1)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    ticks = int(round((y_hi - y_lo) / y_step))
    for k in range(ticks + 1):
        v = y_lo + k * y_step
        parts.append(f'<line class="{"baseline" if abs(v) < 1e-9 else "gridline"}" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{_e(y_fmt.format(v))}</text>')
    for k in range(6):
        v = x_lo + (x_hi - x_lo) * k / 5
        parts.append(f'<text x="{sx(v):.1f}" y="{h - bottom + 16}" text-anchor="middle">{_e(x_fmt.format(v))}</text>')
    if y_lo < 0 < y_hi:
        parts.append(f'<line class="baseline" x1="{left}" x2="{w - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    if diagonal:
        parts.append(f'<line class="ref" x1="{sx(x_lo):.1f}" y1="{sy(y_lo):.1f}" x2="{sx(x_hi):.1f}" y2="{sy(y_hi):.1f}"/>')
    if not dots and len(points) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(points))
        parts.append(f'<path class="line" d="{path}"/>')
    if dots:
        for p in points:
            label = p[2] if len(p) > 2 else f"{x_fmt.format(p[0])}, {y_fmt.format(p[1])}"
            parts.append(f'<circle class="dot" cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="5" tabindex="0" '
                         f'data-value="{_e(y_fmt.format(p[1]))}" data-label="{_e(label)}" aria-label="{_e(label)}"/>')
    if marker:
        parts.append(f'<circle class="dot" cx="{sx(marker[0]):.1f}" cy="{sy(marker[1]):.1f}" r="6" tabindex="0" '
                     f'data-value="Chosen threshold" data-label="{_e(marker_label)}" aria-label="{_e(marker_label)}"/>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text>'
                 f'<text class="label" x="12" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {top + ph / 2})">{_e(y_title)}</text></svg>')
    return "".join(parts)


def _table(headers, rows, numeric=()):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(1,)):
    return f'<details class="tv"><summary>Show as table</summary><div class="scroll">{_table(headers, rows, numeric)}</div></details>'


def _tile(label, value, note="", hero=False):
    return (f'<div class="card tile{" hero" if hero else ""}"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            + (f'<div class="note">{_e(note)}</div>' if note else "") + "</div>")


def render_html(meta):
    if "gains" not in meta:
        raise ModelError("This model was saved without test results; train it again to build a report.")
    test, cv, rows, eco = meta["metrics"]["test"], meta["metrics"]["cv"], meta["rows"], meta["economics"]
    model, everyone = _policy(meta, "Use the model"), _policy(meta, "Contact everybody")
    body = [f'<header><h1>Customer Churn Prediction — model report</h1><div class="sub">{_e(meta["model"])}'
            f'{" (" + _e(meta["model_note"]) + ")" if meta.get("model_note") else ""} · trained {_e(meta["trained_at"].replace("T", " "))} · '
            f'data: {_e(meta["source"])} · {rows["raw"]} rows read, {rows["clean"]} after cleaning, {rows["train"]} train / {rows["test"]} test</div></header>']
    body.append('<section aria-label="Summary"><div class="tiles">' + "".join([
        _tile("ROC AUC", f"{test['roc_auc']:.3f}", "0.5 would be a coin flip", hero=True),
        _tile("PR AUC", f"{test['average_precision']:.3f}", f"guessing scores {test['churn_rate'] / 100:.3f}"),
        _tile("Brier score", f"{test['brier']:.3f}", "honesty of probabilities, lower is better"),
        _tile("Churners flagged", f"{test['recall']:.0f}%", f"at the {test['threshold'] * 100:.1f}% contact threshold"),
        _tile("Flags that were right", f"{test['precision']:.0f}%", f"{test['flagged']} customers flagged"),
        _tile("Campaign profit", money(model["profit"]), f"offering to everybody: {money(everyone['profit'])}")]) + "</div></section>")
    body.append('<section class="card"><h2>What this means</h2><ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in _verdict(meta)) + "</ul></section>")

    comp = [(c["model"], round(c["roc_auc"], 3)) for c in meta["comparison"]]
    comparison_rows = [[c["model"], f"{c['roc_auc']:.3f}", f"{c['average_precision']:.3f}", f"{c['brier']:.3f}"] for c in meta["comparison"]]
    chosen = next((c["model"] for c in meta["comparison"] if c["model"] == cv["model"]), None)
    share = test["flagged"] / test["n"]
    gains = [(x, y) for x, y in meta["gains"]]
    cal_points = [(r["predicted"] / 100, r["observed"] / 100, f"{r['band']}: predicted {r['predicted']:.0f}%, "
                   f"observed {r['observed']:.0f}% ({r['customers']} customers)") for r in meta["calibration_table"]]
    profit = [(t, v) for t, v in meta["profit_curve"]]
    deciles = [(str(r["decile"]), round(r["churn_rate"])) for r in meta["lift"]]
    imp = [(i["label"], round(i["importance"], 3)) for i in meta["importance"][:8]]
    chosen_x = min(max(eco["contact"], 0), 1)
    chosen_profit = next((v for t, v in profit if abs(t - chosen_x) < 0.011), profit[-1][1])
    gain_label = ("Chosen threshold: contact " + f"{share:.0%}" + " of customers, reach " + f"{test['recall']:.0f}%" + " of churners")
    profit_label = "Chosen threshold " + f"{eco['contact'] * 100:.1f}%"
    cal_rows = [[r["band"], r["customers"], f"{r['predicted']:.0f}%", f"{r['observed']:.0f}%"] for r in meta["calibration_table"]]
    lift_rows = [[r["decile"], f"{r['churn_rate']:.0f}%", f"{r['lift']:.1f}×", f"{r['captured']:.0f}%"] for r in meta["lift"]]
    gain_rows = [[f"{x:.0%}", f"{y:.0%}"] for x, y in gains]
    profit_rows = [[f"{t:.0%}", money(v)] for t, v in profit[::5]]
    gain_chart = xy_chart(gains, "Share of churners reached against share of customers contacted", "Customers contacted",
                          "Churners reached", marker=(share, test["recall"] / 100), marker_label=gain_label, diagonal=True,
                          y_range=(0, 1, 0.2))
    cal_chart = xy_chart(cal_points, "Predicted against observed churn rate", "Predicted churn chance", "Observed churn rate",
                         dots=True, diagonal=True, y_range=(0, 1, 0.2))
    profit_chart = xy_chart(profit, "Campaign profit per 1000 customers against contact threshold",
                            "Contact customers whose churn chance is at least", "Profit per 1,000", y_fmt="{:,.0f}",
                            marker=(chosen_x, chosen_profit), marker_label=profit_label)
    cards = [
        f'<div class="card"><h3>Model comparison</h3><div class="sub">Cross-validated ROC AUC — higher is better</div>'
        f'{hbar_chart(comp, "Cross-validated ROC AUC of each model", chosen, "{:.3f}")}'
        f'{_table_view(["Model", "ROC AUC", "PR AUC", "Brier"], comparison_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>Contacting customers in order of risk</h3><div class="sub">Share of churners reached against share of customers contacted; the dashed line is random contact</div>'
        f'{gain_chart}{_table_view(["Customers contacted", "Churners reached"], gain_rows, (0, 1))}</div>',
        f'<div class="card"><h3>Are the probabilities honest?</h3><div class="sub">Predicted churn chance against what really happened; dots on the dashed line are perfect</div>'
        f'{cal_chart}{_table_view(["Band", "Customers", "Predicted", "Observed"], cal_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>What the threshold is worth</h3><div class="sub">Expected campaign profit per 1,000 customers for each contact threshold</div>'
        f'{profit_chart}{_table_view(["Threshold", "Profit per 1,000"], profit_rows, (0, 1))}</div>',
        f'<div class="card"><h3>Churn rate by risk decile</h3><div class="sub">Customers sorted from most to least at risk, cut into ten equal groups (1 = most at risk)</div>'
        f'{column_chart(deciles, "Churn rate by risk decile", "Risk decile", "% churned")}'
        f'{_table_view(["Decile", "Churn rate", "Lift", "Churners caught so far"], lift_rows, (0, 1, 2, 3))}</div>',
        f'<div class="card"><h3>What the model relies on</h3><div class="sub">Drop in ROC AUC when a feature is scrambled</div>'
        f'{hbar_chart(imp, "Feature importance", None, "{:.3f}")}{_table_view(["Feature", "AUC drop"], imp)}</div>',
    ]
    body.append(f'<section><h2>Charts</h2><div class="grid">{"".join(cards)}</div></section>')

    matrix = _table(["", "Flagged for an offer", "Not flagged"],
                    [["Really left", test["tp"], f"{test['fn']} (missed)"],
                     ["Really stayed", f"{test['fp']} (offer wasted)", test["tn"]]], (1, 2))
    policy_rows = [[p["policy"], p["contacted"], p["churners_reached"], f"{p['saved']:.0f}", money(p["profit"])] for p in meta["policies"]]
    policy_table = _table(["Policy", "Contacted", "Churners reached", "Expected saves", "Expected profit"], policy_rows, (1, 2, 3, 4))
    seg_head = ["Customers", "Really left", "Predicted", "Gap"]

    def seg(key):
        return [[s["group"], s["n"], f"{s['actual']:.1f}%", f"{s['predicted']:.1f}%", f"{s['gap']:+.1f}"] for s in meta["segments"][key]]
    contract_table = _table(["Contract"] + seg_head, seg("contract"), (1, 2, 3, 4))
    internet_table = _table(["Internet"] + seg_head, seg("internet"), (1, 2, 3, 4))
    payment_table = _table(["Payment"] + seg_head, seg("payment"), (1, 2, 3, 4))
    body.append('<section><h2>Money and fairness checks</h2><div class="grid">'
                f'<div class="card"><h3>What a retention campaign earns</h3><div class="sub">Offer cost {money(eco["cost"])}, customer value {money(eco["value"])}, '
                f'{eco["save_rate"] * 100:.0f}% of offers work: an offer pays off above a {eco["break_even"] * 100:.1f}% churn chance. Test customers.</div>{policy_table}</div>'
                f'<div class="card"><h3>Confusion matrix</h3>{matrix}</div>'
                f'<div class="card"><h3>By contract</h3>{contract_table}</div>'
                f'<div class="card"><h3>By internet service</h3>{internet_table}</div>'
                f'<div class="card"><h3>By payment method</h3>{payment_table}</div></div></section>')

    steps = meta["cleaning"]["steps"]
    body.append('<section class="card"><h2>Data cleaning log</h2><div class="sub">'
                f'{meta["cleaning"]["rows_in"]} rows read → {meta["cleaning"]["rows_out"]} kept</div>'
                + _table(["Problem found", "Rows", "Action"], [[s["label"], s["count"], s["outcome"]] for s in steps], (1,))
                + "</section>")
    params = ", ".join(f"{k} = {v}" for k, v in meta.get("model_params", {}).items()) or "defaults"
    env = meta["environment"]
    body.append(f'<footer>Generated {datetime.now():%d %b %Y, %H:%M} · settings: {_e(params)} · seed {meta["seed"]} · '
                f'Python {_e(env["python"])}, scikit-learn {_e(env["sklearn"])} · developed by Sneha S · '
                'Trained on synthetic customers: for learning only, real-world accuracy would differ.</footer>')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Customer churn model report</title><style>{CSS}</style></head><body class='rep'><main>" + "\n".join(body)
            + f"</main><div id='tip' role='status' aria-live='polite'></div><script>{TOOLTIP_JS}</script></body></html>")


def save_html(meta, path):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(meta), encoding="utf-8")
    except OSError as exc:
        raise ModelError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path
