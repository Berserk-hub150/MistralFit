"""Offline HTML report. User-controlled strings are always escaped."""

import html
import json


def render(report: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    maximum = max((r.get("metrics", {}).get("median_decode_tokens_per_second", 0)
                   for r in report["results"]), default=1) or 1
    for entry in report["results"]:
        metrics, name = entry.get("metrics", {}), entry["profile"]["name"]
        speed = metrics.get("median_decode_tokens_per_second")
        ttft = metrics.get("median_ttft_seconds")
        pct = 100 * speed / maximum if speed is not None else 0
        rows.append(f'<tr><td>{esc(name)}{" ★" if name == report.get("winner") else ""}</td>'
                    f'<td>{esc(entry["status"])}</td><td>{speed:.2f}<div class="track"><div style="width:{pct:.2f}%"></div></div></td>'
                    f'<td>{ttft:.3f}s</td><td>{esc(entry.get("reason", ""))}</td></tr>' if speed is not None else
                    f'<tr><td>{esc(name)}</td><td>{esc(entry["status"])}</td><td>—</td><td>—</td><td>{esc(entry.get("reason", ""))}</td></tr>')
    settings = report["settings"]
    ratio = report.get("speedup_vs_upstream_auto")
    comparison = f'{ratio:.3f}× measured decode throughput versus upstream auto' if ratio else "No valid upstream comparison available"
    consistency = report.get("winner_matches_baseline_output")
    check = "Matching output hashes" if consistency is True else "Different output hashes — inspect correctness before adopting" if consistency is False else "Output comparison unavailable"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MistralFit experiment</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#11151a;color:#e5ebef;font:16px system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:52px 28px}}.tag{{color:#f7b955;font-size:13px;letter-spacing:2px}}h1{{font-size:46px;margin:14px 0}}
p{{color:#aab8c4;line-height:1.6}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:30px 0}}.card{{background:#1c232c;padding:22px;border-radius:12px;flex:1;min-width:180px}}
.card strong{{display:block;font-size:23px;margin-top:10px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:15px 10px;text-align:left;border-bottom:1px solid #303943}}
th{{color:#aab8c4;font-size:13px}}td:last-child{{font-size:13px;color:#aab8c4}}.track{{background:#303943;height:5px;border-radius:4px;margin-top:8px}}.track div{{height:5px;background:#f7b955;border-radius:4px}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c232c;padding:22px;border-radius:12px}}.notice{{border-left:3px solid #f7b955;padding-left:18px}}
</style><main><div class="tag">MISTRALFIT · LOCAL INFERENCE EXPERIMENT</div><h1>Measure. Compare. Choose.</h1>
<p>{esc(report['plan']['model']['name'])} · {esc(report['created_at'])} · {esc(report['status'])}</p>
<div class="cards"><div class="card">Best measured candidate<strong>{esc(report.get('winner') or 'None')}</strong></div>
<div class="card">Fixed context<strong>{esc(settings['context'])} tokens</strong></div><div class="card">Measured requests / profile<strong>{esc(settings['repeats'])}</strong></div></div>
<p>{esc(comparison)}. {esc(check)}.</p><div class="scroll"><table><thead><tr><th>Placement</th><th>Status</th><th>Median decode tok/s</th><th>Median TTFT</th><th>Notes</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="notice">These measurements belong to this model, runtime, hardware and workload. No language-quality evaluation or peak-memory measurement was performed. Estimates are separate from measurements.</p>
<h2>Method and reproducibility</h2><pre>{esc(json.dumps(settings, indent=2))}</pre>
<h2>Limits</h2><ul>{''.join('<li>'+esc(x)+'</li>' for x in report.get('limits', []))}</ul>
<h2>Runtime</h2><pre>{esc(report.get('runtime', {}).get('version', 'unknown'))}</pre>
<p>Generated offline. No analytics, remote fonts, JavaScript, or network requests.</p></main></html>'''
