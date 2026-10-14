"""Local web interface for predictions (standard library only).

    GET  /              the form
    POST /              the form submitted → result page
    POST /api/predict   JSON in, JSON out
    GET  /api/health    model information

Safety measures: binds to 127.0.0.1 only; the Host header must be a local name (blocks DNS-rebinding);
request bodies are size-limited; every value is validated by the same code as the command line;
everything shown is HTML-escaped; the page has no scripts and a strict Content-Security-Policy.
"""

import html
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import schema as S
from .exceptions import ChurnError, ValidationError

log = logging.getLogger("churnpredict")

MAX_BODY = 20_000
DRAIN_LIMIT = 1_000_000   # a rejected body up to this size is read and discarded before the error is sent
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                               "base-uri 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}
FORM_DEFAULTS = {"tenure_months": "12", "monthly_charges": "75", "total_charges": "", "support_calls": "1",
                 "late_payments": "0", "contract": "Month-to-month", "internet_service": "Fiber",
                 "payment_method": "Bank transfer (auto)", "paperless_billing": "no", "tech_support": "no",
                 "online_security": "no"}
TIER_CLASS = {"High": "risk", "Medium": "watch", "Low": "ok"}

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--border:rgba(11,11,11,.12);--err:#b3261e;--errbg:#fdecea;--warnbg:#fff4e0;--okbg:#e6f4ea;--ok:#1e6b34}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--border:rgba(255,255,255,.12);--err:#ff8a80;--errbg:#3d1717;--warnbg:#3a2c12;
--okbg:#16301d;--ok:#8fd9a0}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 8px}h3{font-size:14px;margin:16px 0 6px}
.sub{color:var(--ink2);font-size:13px}.badge{font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 8px;margin-left:8px;color:var(--ink2)}
.layout{display:grid;grid-template-columns:minmax(300px,400px) 1fr;gap:16px;margin-top:20px;align-items:start}
@media (max-width:760px){.layout{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px 12px}label{display:block;font-size:13px;color:var(--ink2)}
.full{grid-column:1/-1}input,select{width:100%;margin-top:3px;padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--page);
color:var(--ink);font:inherit}input:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
.err{color:var(--err);font-size:12px;margin-top:2px}.invalid input,.invalid select{border-color:var(--err)}
button{grid-column:1/-1;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.hero{font-size:48px;font-weight:600;line-height:1.1;margin:4px 0}.hero small{font-size:16px;color:var(--ink2);font-weight:400}
.chip{display:inline-block;font-size:13px;font-weight:600;border-radius:999px;padding:3px 12px;border:1px solid var(--border)}
.chip.ok{background:var(--okbg);color:var(--ok)}.chip.watch{background:var(--warnbg)}.chip.risk{background:var(--errbg);color:var(--err)}
.meter{position:relative;height:10px;border-radius:5px;background:var(--line);margin:16px 0 4px}
.meter i{position:absolute;top:0;bottom:0;left:0;border-radius:5px;background:var(--accent);opacity:.45}
.meter u{position:absolute;top:-5px;width:2px;height:20px;background:var(--ink2);text-decoration:none}
.meterlab{position:relative;height:16px;font-size:12px;color:var(--muted)}.meterlab span{position:absolute;white-space:nowrap}
.offer{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:12px 0;font-size:14px}
.factor{display:grid;grid-template-columns:200px 1fr 90px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.factor .track{height:10px;background:var(--line);border-radius:5px;position:relative}.factor .fill{position:absolute;top:0;bottom:0;border-radius:5px;background:var(--accent)}
.factor .fill.neg{background:var(--muted)}.factor .amt{text-align:right;font-variant-numeric:tabular-nums}
ul.advice{margin:6px 0;padding-left:20px;font-size:14px}ul.advice li{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line)}th{color:var(--ink2)}
.num{text-align:right;font-variant-numeric:tabular-nums}.note{background:var(--warnbg);border-radius:8px;padding:8px 10px;margin:8px 0;font-size:13px}
.error{background:var(--errbg);color:var(--err);border-radius:8px;padding:10px;margin-bottom:12px}
footer{margin-top:24px;color:var(--muted);font-size:12px}@media (max-width:520px){.factor{grid-template-columns:1fr 70px}.factor .track{display:none}}
"""


def e(value):
    return html.escape(str(value), quote=True)


def select(name, options, current, blank=None):
    items = ([f'<option value=""{" selected" if not current else ""}>{e(blank)}</option>'] if blank else [])
    items += [f'<option{" selected" if o == current else ""}>{e(o)}</option>' for o in options]
    return f'<select name="{name}" id="{name}">{"".join(items)}</select>'


def number_input(name, current, required=True):
    kind, low, high, _ = S.NUMERIC[name]
    step = "1" if kind == "int" else "0.01"
    return (f'<input type="number" name="{name}" id="{name}" min="{low}" max="{high}" step="{step}" '
            f'value="{e(current)}" inputmode="decimal"{" required" if required else ""}>')


def field(name, control, errors, wide=False):
    label = S.LABELS[name] + (f" ({S.UNITS[name]})" if name in S.UNITS else "")
    message = f'<div class="err">{e(errors[name])}</div>' if name in errors else ""
    return f'<div class="{"full " if wide else ""}{"invalid" if name in errors else ""}"><label for="{name}">{e(label)}{control}</label>{message}</div>'


def form_html(values, errors):
    v = {**FORM_DEFAULTS, **values}
    parts = [
        field("contract", select("contract", S.CONTRACTS, v["contract"]), errors),
        field("internet_service", select("internet_service", S.INTERNET, v["internet_service"]), errors),
        field("tenure_months", number_input("tenure_months", v["tenure_months"]), errors),
        field("monthly_charges", number_input("monthly_charges", v["monthly_charges"]), errors),
        field("total_charges", number_input("total_charges", v["total_charges"], required=False), errors),
        field("support_calls", number_input("support_calls", v["support_calls"]), errors),
        field("late_payments", number_input("late_payments", v["late_payments"]), errors),
        field("payment_method", select("payment_method", S.PAYMENTS, v["payment_method"]), errors),
        field("paperless_billing", select("paperless_billing", ["no", "yes"], v["paperless_billing"]), errors),
        field("tech_support", select("tech_support", ["no", "yes"], v["tech_support"]), errors),
        field("online_security", select("online_security", ["no", "yes"], v["online_security"]), errors),
    ]
    return (f'<form method="post" action="/" novalidate>{"".join(parts)}'
            '<div class="sub full">Leave total charges blank to use tenure × monthly charges.</div>'
            '<button type="submit">Predict churn</button></form>')


def result_html(p, meta):
    eco, t = meta["economics"], meta["thresholds"]["contact"]
    pct = p.probability * 100
    watch, contact = t * 50, min(t, 1) * 100
    if p.contact:
        offer = (f"<b>Make a retention offer.</b> Expected gain about {e(S.format_money(p.expected_gain))} per customer "
                 f"(an offer pays off above a {eco['break_even'] * 100:.1f}% churn chance).")
    else:
        offer = (f"<b>No offer needed.</b> An offer would cost more than it is expected to save "
                 f"({e(S.format_money(p.expected_gain))} per customer; break-even is a {eco['break_even'] * 100:.1f}% churn chance).")
    out = [f'<h2>Churn risk</h2><span class="chip {TIER_CLASS[p.tier]}">{e(p.tier)} risk</span>',
           f'<div class="hero">{pct:.0f}%<small> chance of leaving</small></div>',
           f'<div class="meter" aria-hidden="true"><i style="width:{pct:.1f}%"></i><u style="left:{watch:.1f}%"></u>'
           f'<u style="left:{min(contact, 100):.1f}%"></u></div>'
           f'<div class="meterlab"><span style="left:0">0%</span><span style="left:{min(watch, 70):.1f}%">medium from {watch:.1f}%</span>'
           f'<span style="right:0">high from {contact:.1f}%</span></div>',
           f'<div class="offer">{offer}</div>']
    for w in p.warnings:
        out.append(f'<div class="note">⚠ {e(w)}</div>')
    if p.actions:
        items = "".join(f'<li>{e(a["action"])} — churn chance {pct:.0f}% → {a["new_probability"] * 100:.0f}% '
                        f'(<b>−{a["reduction_points"]:.0f} points</b>, about {e(S.format_money(a["retained_value"]))} of kept revenue)</li>'
                        for a in p.actions)
        out.append(f"<h3>What could keep this customer</h3><ul class='advice'>{items}</ul>"
                   "<div class='sub'>Estimates from patterns in the data, not guarantees.</div>")
    if p.factors:
        top = max(abs(f["effect_points"]) for f in p.factors) or 1
        out.append("<h3>What drives this risk (compared with a typical customer)</h3>")
        for f in p.factors:
            width = max(3, 100 * abs(f["effect_points"]) / top)
            sign = "+" if f["effect_points"] >= 0 else "−"
            out.append(f'<div class="factor"><span>{e(f["label"])}: {e(f["value"])}</span>'
                       f'<div class="track"><div class="fill{" neg" if f["effect_points"] < 0 else ""}" style="width:{width:.0f}%"></div></div>'
                       f'<span class="amt">{sign}{abs(f["effect_points"]):.0f} points</span></div>')
        out.append('<div class="sub">“+” raises the churn chance compared with the typical value, “−” lowers it.</div>')
    if p.similar:
        def cell(value):
            return "—" if value is None else value
        rows = "".join(f'<tr><td>{e(c["contract"])}</td><td>{e(c["internet_service"])}</td><td class="num">{cell(c["tenure_months"])}</td>'
                       f'<td class="num">{cell(c["monthly_charges"])}</td><td class="num">{cell(c["support_calls"])}</td>'
                       f'<td><b>{e(c["outcome"])}</b></td></tr>' for c in p.similar)
        out.append("<h3>Similar customers in the training data</h3><table><thead><tr><th>Contract</th><th>Internet</th>"
                   "<th class='num'>Tenure</th><th class='num'>Monthly</th><th class='num'>Calls</th><th>Outcome</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table>")
    return "".join(out)


def page(meta, values=None, errors=None, prediction=None, message=None):
    errors = errors or {}
    test = meta["metrics"]["test"]
    if prediction is not None:
        right = result_html(prediction, meta)
    elif message:
        right = f'<div class="error">{e(message)}</div>'
    else:
        right = ('<h2>Describe a customer</h2><p class="sub">Fill in the account details on the left. The model gives the chance '
                 'the customer leaves, says whether a retention offer is worth making, and shows what could keep them.</p>'
                 f'<p class="sub">Model: {e(meta["model"])} · ROC AUC {test["roc_auc"]:.3f} on unseen customers · '
                 f'the offer threshold is a {meta["thresholds"]["contact"] * 100:.1f}% churn chance</p>')
    summary = e(errors["_"]) if "_" in errors else ""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Customer Churn Prediction</title><style>{CSS}</style></head><body><main>"
            "<h1>Customer Churn Prediction<span class='badge'>demo · synthetic data</span></h1>"
            "<div class='sub'>Predicts which customers are likely to leave, so a retention team can act first.</div>"
            f"<div class='layout'><div class='card'>{('<div class=error>' + summary + '</div>') if summary else ''}"
            f"{form_html(values or {}, errors)}</div><div class='card' aria-live='polite'>{right}</div></div>"
            "<footer>Developed by Sneha S · Trained on generated data to learn the machine-learning workflow — "
            "a guide for offering help, never a reason to treat a real customer unfairly.</footer></main></body></html>")


def make_server(predictor, host="127.0.0.1", port=8030):
    meta = predictor.meta

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChurnPredict/1.0"

        def log_message(self, fmt, *args):   # keep the terminal quiet; real problems go to the log file
            log.info("web: " + fmt, *args)

        # ----- helpers
        def send(self, status, body, content_type="text/html; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in SECURITY_HEADERS.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, payload):
            self.send(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

        def host_ok(self):
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[:host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in LOCAL_HOSTS

        def read_body(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ValidationError("Content-Length header is required.") from None
            if length < 0 or length > MAX_BODY:
                self.discard_body(length)
                raise OverflowError
            return self.rfile.read(length)

        def discard_body(self, length):
            """Read and throw away (a bounded amount of) a rejected body before replying.

            If the server answers and closes while the client is still sending, the operating system may
            reset the connection and the client never sees the 413 message. Huge bodies are not drained
            (that would let one request tie the server up); for those the connection is simply closed.
            """
            remaining = min(max(length, 0), DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True

        def guard(self):
            if not self.host_ok():
                self.send(403, "Forbidden: this server only answers to localhost.", "text/plain; charset=utf-8")
                return False
            return True

        # ----- routes
        def do_GET(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, page(meta))
            elif path == "/api/health":
                test = meta["metrics"]["test"]
                self.send_json(200, {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
                                     "test_roc_auc": round(test["roc_auc"], 4),
                                     "contact_threshold": round(meta["thresholds"]["contact"], 4),
                                     "features": S.RAW_FEATURES})
            else:
                self.send(404, page(meta, message="Page not found."))

        def do_POST(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path not in ("/", "/api/predict"):
                return self.send(404, page(meta, message="Page not found."))
            api = path == "/api/predict"
            try:
                body = self.read_body()
            except OverflowError:
                return self.send_json(413, {"error": "Request too large."}) if api else \
                    self.send(413, page(meta, message="That request is too large."))
            except ValidationError as exc:
                return self.send_json(411, {"error": str(exc)}) if api else self.send(411, page(meta, message=str(exc)))
            try:
                if api:
                    if "application/json" not in (self.headers.get("Content-Type") or ""):
                        return self.send_json(415, {"error": "Send JSON with Content-Type: application/json."})
                    try:
                        raw = json.loads(body.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return self.send_json(400, {"error": "The request body is not valid JSON."})
                    return self.send_json(200, predictor.predict(raw).to_dict())
                form = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=30)
                raw = {k: v[0] for k, v in form.items()}
                try:
                    prediction = predictor.predict(raw)
                except ValidationError as exc:
                    return self.send(422, page(meta, raw, exc.errors or {"_": str(exc)}))
                self.send(200, page(meta, raw, prediction=prediction))
            except ValidationError as exc:
                self.send_json(422, {"error": str(exc), "fields": exc.errors})
            except ChurnError as exc:
                self.send_json(500, {"error": str(exc)}) if api else self.send(500, page(meta, message=str(exc)))
            except Exception:  # noqa: BLE001 - never leak internals to the browser
                log.exception("Unexpected error handling %s", path)
                self.send_json(500, {"error": "Something went wrong."}) if api else \
                    self.send(500, page(meta, message="Something went wrong. See the log file."))

    return ThreadingHTTPServer((host, port), Handler)


def serve(predictor, port=8030, host="127.0.0.1"):
    server = make_server(predictor, host, port)
    print(f"  Customer Churn Prediction is running at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Web interface stopped.")
    finally:
        server.server_close()
