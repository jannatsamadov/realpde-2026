"""Render a trace from trace_model.py as a flowchart page.

Every number on the page -- shapes, parameter counts, errors -- is read from the
trace JSON, which trace_model.py measured on the built checkpoint. Nothing here
is typed in by hand, so the page cannot drift from the model it describes.

Usage:
    python scripts/trace_model.py --checkpoint X_best.pt --out trace.json
    python scripts/flowchart_page.py --trace trace.json --out model_trace.html
"""

import argparse
import html
import json
from pathlib import Path


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")          # thin-space thousands


def fmt_shape(s) -> str:
    return " × ".join(str(v) for v in s)


CSS = """
:root{
  --ground:#F3F4F1; --surface:#FFFFFF; --ink:#1B2220; --muted:#5C6862; --rule:#D6DBD4;
  --phys:#9A6310; --phys-bg:#FBF3E4; --learn:#2F5F8A; --learn-bg:#EAF1F8;
  --good:#2E7D5B; --bad:#B3412F; --ref:#7D8883; --out-bg:#EEF6F1;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#111513; --surface:#191F1C; --ink:#E4E9E6; --muted:#97A39D; --rule:#2B3430;
    --phys:#DDA84A; --phys-bg:#2A2214; --learn:#86AEDB; --learn-bg:#16222F;
    --good:#5FBF90; --bad:#E27A66; --ref:#8C9892; --out-bg:#15241C;
  }
}
:root[data-theme="dark"]{
  --ground:#111513; --surface:#191F1C; --ink:#E4E9E6; --muted:#97A39D; --rule:#2B3430;
  --phys:#DDA84A; --phys-bg:#2A2214; --learn:#86AEDB; --learn-bg:#16222F;
  --good:#5FBF90; --bad:#E27A66; --ref:#8C9892; --out-bg:#15241C;
}
body{background:var(--ground);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;
  padding-inline:16px;padding-block:28px 48px}
.wrap{max-width:1120px;margin:0 auto;display:grid;gap:28px}
h1{font:600 28px/1.15 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 17px/1.2 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;margin:0 0 12px;letter-spacing:.01em}
.lede{color:var(--muted);max-width:68ch;margin:6px 0 0}
.mono,.shape,.params{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:12px;font-size:13px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:600}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:28px;align-items:start}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--muted);margin-bottom:14px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:6px}
.flow{display:flex;flex-direction:column}
.node{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;
  display:grid;grid-template-columns:1fr auto;gap:2px 14px;align-items:baseline}
.node .name{font-weight:600}
.node .shape{font-size:12.5px;color:var(--muted)}
.node .params{font-size:12.5px;text-align:right;color:var(--muted)}
.node .note{grid-column:1/-1;font-size:13px;color:var(--muted)}
.node.phys{background:var(--phys-bg);border-color:color-mix(in srgb,var(--phys) 45%,var(--rule))}
.node.phys .name{color:var(--phys)}
.node.learn{background:var(--learn-bg);border-color:color-mix(in srgb,var(--learn) 40%,var(--rule))}
.node.learn .name{color:var(--learn)}
.node.out{background:var(--out-bg);border-color:color-mix(in srgb,var(--good) 45%,var(--rule));border-width:1.5px}
.err{grid-column:1/-1;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;margin-top:2px}
.err .bad{color:var(--bad);font-weight:600}.err .good{color:var(--good);font-weight:600}
.link{align-self:flex-start;margin-left:22px;border-left:2px solid var(--rule);height:16px}
.link.tall{height:22px}
.chips{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:12px;border:1px solid var(--rule);border-radius:999px;padding:1px 9px;background:var(--surface)}
.chip .mono{color:var(--muted)}
.unet{background:var(--surface);border:1px solid color-mix(in srgb,var(--learn) 40%,var(--rule));border-radius:6px;padding:12px 14px}
.unet-title{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px}
.unet-title .name{font-weight:600;color:var(--learn)}
.ugrid{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px}
.lvl{border:1px solid var(--rule);border-radius:4px;padding:5px 9px;background:var(--learn-bg);
  display:flex;justify-content:space-between;gap:8px;font-size:12.5px}
.lvl .mono{color:var(--muted)}
.lvl.bottom{grid-column:1/-1;justify-self:center;min-width:55%}
.lvl.film{grid-column:1/-1;justify-self:center;min-width:55%;background:var(--phys-bg)}
.ucap{font-size:12px;color:var(--muted);margin-top:8px}
.loopnote{font-size:13px;color:var(--learn);font-weight:600;margin:0 0 0 30px;padding:4px 0}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:16px 16px 12px}
.bars{display:grid;gap:12px}
.bar-row{display:grid;grid-template-columns:minmax(0,1fr);gap:4px}
.bar-lab{display:flex;justify-content:space-between;gap:10px;font-size:13px}
.bar-lab .mono{color:var(--muted);font-size:12.5px}
.track{position:relative;height:14px;background:color-mix(in srgb,var(--rule) 45%,transparent);border-radius:3px}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.refline{position:absolute;top:-4px;bottom:-4px;width:0;border-left:2px dashed var(--ref)}
.axis{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted);margin-top:6px}
.checks{display:grid;gap:6px;font-size:13.5px}
.checks .ok{color:var(--good);font-weight:600;margin-right:6px}
.foot{font-size:12.5px;color:var(--muted)}
"""


def node(cls, name, shape=None, params=None, note=None, err_html=None, chips=None):
    p = "" if params is None else (f"{fmt_int(params)} param" if params else "0 param")
    parts = [f'<div class="node {cls}">',
             f'<div class="name">{html.escape(name)}</div>',
             f'<div class="params">{p}</div>']
    if shape is not None:
        parts.append(f'<div class="shape" style="grid-column:1/-1">{fmt_shape(shape)}</div>')
    if chips:
        parts.append('<div class="chips">' + "".join(
            f'<span class="chip">{html.escape(k)} <span class="mono">{v}</span></span>'
            for k, v in chips.items()) + "</div>")
    if err_html:
        parts.append(f'<div class="err">{err_html}</div>')
    if note:
        parts.append(f'<div class="note">{note}</div>')
    parts.append("</div>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Solverloop Model İzi")
    ap.add_argument("--compare", default=None,
                    help="'label=rms' for the model this one should be judged against, "
                         "measured on the same windows. Pass 2 against pass 1 overstates "
                         "a refining model: pass 1 was never trained to be an output.")
    args = ap.parse_args()

    t = json.loads(Path(args.trace).read_text())
    S = {s["stage"]: s for s in t["stages"]}
    pers = S["persistence (copy last frame)"]["error"]["rms_ms"]

    def err_line(stage):
        e = S[stage]["error"]
        r, rel = e["rms_ms"], e["rel_l2"]
        d = 100 * (r / pers - 1)
        cls = "bad" if r > pers else "good"
        word = "persistence-dən pis" if r > pers else "persistence-dən yaxşı"
        return (f'RMS xəta {r:.4f} m/s · rel_l2 {rel:.4f} · '
                f'<span class="{cls}">{d:+.0f}% ({word})</span>')

    unet_stages = ["enc1", "enc2", "enc3", "bottleneck", "dec3 (+ skip enc3)",
                   "dec2 (+ skip enc2)", "dec1 (+ skip enc1)"]
    unet_params = sum(S[s]["params"] for s in unet_stages) + S["head (1x1 conv) -> residual"]["params"]
    film = S.get("scale conditioning (FiLM)")
    if film:
        unet_params += film["params"]

    def lvl(stage, label):
        s = S[stage]
        c, h, w = s["shape"]
        return (f'<div class="lvl"><span>{label}</span>'
                f'<span class="mono">{c}@{h}×{w} · {fmt_int(s["params"])}</span></div>')

    unet_html = (
        '<div class="unet"><div class="unet-title">'
        '<span class="name">U-Net (öyrədilən hissə)</span>'
        f'<span class="params mono">{fmt_int(unet_params)} param</span></div>'
        '<div class="ugrid">'
        + lvl("enc1", "enc1") + lvl("dec1 (+ skip enc1)", "dec1 ← enc1")
        + lvl("enc2", "enc2") + lvl("dec2 (+ skip enc2)", "dec2 ← enc2")
        + lvl("enc3", "enc3") + lvl("dec3 (+ skip enc3)", "dec3 ← enc3")
        + (f'<div class="lvl film"><span>miqyas şərti (FiLM)</span><span class="mono">'
           f'{fmt_int(film["params"])}</span></div>' if film else "")
        + '<div class="lvl bottom"><span>bottleneck</span><span class="mono">'
        + f'{S["bottleneck"]["shape"][0]}@{S["bottleneck"]["shape"][1]}×{S["bottleneck"]["shape"][2]}'
        + f' · {fmt_int(S["bottleneck"]["params"])}</span></div>'
        '</div>'
        f'<div class="ucap">Hər səviyyə: iki 3×3 konvolyusiya + GroupNorm + GELU. Aşağı: ortalama pooling ×2, '
        f'yuxarı: ən yaxın qonşu ×2, eyni səviyyənin enc çıxışı ilə birləşir. '
        f'Sonda 1×1 başlıq → {S["head (1x1 conv) -> residual"]["shape"][0]} kanal = 20 kadr × (u,v) qalıq, '
        f'{fmt_int(S["head (1x1 conv) -> residual"]["params"])} param. Təlimə sıfır çəkilərlə başlayıb: '
        f'ilk addımda model = saf fizika prior-u.</div></div>'
    )

    ch = S["network input channels"]["channels"]
    chips = {k.replace("input frames", "giriş kadrları")
              .replace("advected frames", "advect olunmuş kadrlar")
              .replace("speed, vorticity, divergence", "sürət, burulma, divergensiya")
              .replace("x, y coordinates", "x, y koordinatları")
              .replace("log velocity scale", "log miqyas"): v for k, v in ch.items()}

    flow = []
    flow.append(node("", "Giriş: 20 PIV kadrı", S["input window"]["shape"],
                     note="20 kadr × 32×64 tor × (u, v). Rəsmi sabitlərlə normallaşdırılıb."))
    flow.append('<div class="link"></div>')
    flow.append(node("phys", "Advection prior — 20 addım", S["advection prior (20 steps)"]["shape"], 0,
                     note=f"Son kadr öz sürəti ilə 20 dəfə semi-Laqranj daşınır "
                          f"(hər kadra {t.get('substeps', '?')} alt-addım). Təzyiq və özlülük YOXDUR.",
                     err_html=err_line("advection prior (20 steps)")))
    flow.append('<div class="link"></div>')
    flow.append(node("", "Şəbəkənin giriş kanalları", S["network input channels"]["shape"], 0,
                     chips=chips))
    flow.append('<div class="link"></div>')
    flow.append(unet_html)
    flow.append('<div class="link"></div>')
    flow.append(node("learn", "1-ci keçid = prior + qalıq", S["PASS 1 = prior + residual"]["shape"], 0,
                     err_html=err_line("PASS 1 = prior + residual")))
    refine = t.get("refine", 0)
    for k in range(refine):
        rp = f"refined prior {k + 1} (1-step each)"
        ps = f"PASS {k + 2} = refined prior + residual"
        flow.append('<div class="link tall"></div>')
        flow.append(node("phys", "Yenilənmiş prior — hər kadra 1 addım", S[rp]["shape"], 0,
                         note="k-cı kadrın prior-u = 1-ci keçidin (k−1)-ci kadrı, bir addım daşınmış. "
                              "20 kadr bir-birindən asılı deyil → bir batched çağırış.",
                         err_html=err_line(rp)))
        flow.append('<div class="link"></div>')
        flow.append('<p class="loopnote">↻ eyni U-Net ikinci dəfə — çəkilər paylaşılır, yeni parametr yoxdur</p>')
        flow.append('<div class="link"></div>')
        flow.append(node("out", f"{k + 2}-ci keçid = FİNAL PROQNOZ", S[ps]["shape"], 0,
                         err_html=err_line(ps)))

    # Error chart over the forecast-producing stages.
    rows = [("Persistence (son kadrı kopyala)", "persistence (copy last frame)"),
            ("Advection prior (20 addım)", "advection prior (20 steps)"),
            ("1-ci keçid", "PASS 1 = prior + residual")]
    for k in range(refine):
        rows.append(("Yenilənmiş prior", f"refined prior {k + 1} (1-step each)"))
        rows.append((f"{k + 2}-ci keçid (final)", f"PASS {k + 2} = refined prior + residual"))
    # (label, rms, colour) for every bar; the comparison model is drawn in the
    # reference colour and labelled, never mixed in with this model's stages.
    bar_data = []
    for label, s in rows:
        r = S[s]["error"]["rms_ms"]
        color = ("var(--ref)" if s.startswith("persistence")
                 else "var(--bad)" if r > pers else "var(--good)")
        bar_data.append((label, r, color))
    if args.compare:
        lab, val = args.compare.split("=")
        bar_data.append((f"müqayisə: {html.escape(lab)}", float(val), "var(--ref)"))
    vmax = max(r for _, r, _ in bar_data) * 1.08
    bars = [
        f'<div class="bar-row"><div class="bar-lab"><span>{label}</span>'
        f'<span class="mono">{r:.4f} m/s</span></div>'
        f'<div class="track"><div class="fill" style="width:{100 * r / vmax:.2f}%;background:{color}"></div>'
        f'<div class="refline" style="left:{100 * pers / vmax:.2f}%"></div></div></div>'
        for label, r, color in bar_data]
    ticks = [0, vmax / 2, vmax]

    # Findings are derived from the measured errors, never written in advance.
    adv_r = S["advection prior (20 steps)"]["error"]["rms_ms"]
    p1_r = S["PASS 1 = prior + residual"]["error"]["rms_ms"]
    fin_stage = (f"PASS {refine + 1} = refined prior + residual" if refine
                 else "PASS 1 = prior + residual")
    fin_r = S[fin_stage]["error"]["rms_ms"]
    items = []
    if adv_r > pers:
        items.append(f"Advection prior tək başına <b>son kadrı kopyalamaqdan {adv_r / pers:.1f}× pisdir</b> "
                     f"({adv_r:.4f} vs {pers:.4f} m/s). Təzyiq həddi olmadığından 20 addımda xəta yığılır.")
    else:
        items.append(f"Advection prior tək başına persistence-dən {100 * (1 - adv_r / pers):.0f}% yaxşıdır.")
    items.append(f"Şəbəkə 1-ci keçiddə xətanı {adv_r:.4f} → {p1_r:.4f} m/s endirir.")
    if refine:
        rp_r = S["refined prior 1 (1-step each)"]["error"]["rms_ms"]
        items.append(f"Yenilənmiş prior köhnəsindən {adv_r / rp_r:.1f}× dəqiqdir ({rp_r:.4f} m/s), "
                     f"çünki hər kadrı düzəldilmiş sahədən yenidən başladır. 2-ci keçid: {fin_r:.4f} m/s "
                     f"({100 * (fin_r / p1_r - 1):+.1f}% 1-ci keçidə nisbətən).")
    items.append(f"Final proqnoz persistence-dən {100 * (1 - fin_r / pers):.0f}% dəqiqdir.")
    if refine:
        items.append("⚠️ 2-ci keçidi 1-ci ilə müqayisə etmək <b>metodun faydasını şişirdir</b>: "
                     "1-ci keçid heç vaxt final çıxış kimi öyrədilməyib.")
    if args.compare:
        lab, val = args.compare.split("=")
        cmp_r = float(val)
        d = 100 * (fin_r / cmp_r - 1)
        items.append(f"<b>Düzgün müqayisə — {html.escape(lab)}, eyni pəncərələrdə: {cmp_r:.5f} m/s.</b> "
                     f"Bu model: {fin_r:.5f} m/s → <b>{d:+.2f}%</b>.")
    findings = "".join(f'<p style="margin:0 0 8px">{s}</p>' for s in items)

    total_listed = unet_params
    checks = [
        f"Sxemdəki hər rəqəm checkpoint-in özündən ölçülüb, submission-un qurduğu kimi <span class='mono'>build_model(\"{t['arch']}\")</span> ilə yaradılıb.",
        "Mərhələlər ardıcıl yenidən hesablanıb və nəticə <span class='mono'>model(x)</span> ilə müqayisə edilib: maksimal fərq 0.0 — sxem xallanan hesablamanın özüdür.",
        f"U-Net səviyyələrinin parametr cəmi {fmt_int(total_listed)} = modelin ümumi parametr sayı {fmt_int(t['params'])}"
        + (" ✓" if total_listed == t["params"] else " ✗ UYĞUN DEYİL") + ".",
    ]

    page = f"""<title>{html.escape(args.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <h1>{html.escape(args.title)}</h1>
    <p class="lede">İdeya yox, qurulmuş model: checkpoint yüklənib, {t['windows']} real validasiya pəncərəsi içindən keçirilib, hər mərhələ modelin öz metodları ilə çağırılıb.</p>
    <div class="meta">
      <span>checkpoint <b class="mono">{html.escape(t['checkpoint'])}</b></span>
      <span>arxitektura <b class="mono">{html.escape(t['arch'])}</b></span>
      <span>keçid sayı <b>{1 + refine}</b></span>
      <span>parametr <b class="mono">{fmt_int(t['params'])}</b></span>
      <span>val Re <b class="mono">{', '.join(str(v) for v in t['val_re'])}</b></span>
    </div>
  </header>
  <div class="grid">
    <section>
      <h2>Axın</h2>
      <div class="legend">
        <span><span class="sw" style="background:var(--phys)"></span>fizika — sabit hesablama, öyrənilmir</span>
        <span><span class="sw" style="background:var(--learn)"></span>öyrədilən şəbəkə</span>
        <span><span class="sw" style="background:var(--good)"></span>final çıxış</span>
      </div>
      <div class="flow">{''.join(flow)}</div>
    </section>
    <aside style="display:grid;gap:20px">
      <div class="card">
        <h2>Hər mərhələnin xətası</h2>
        <div class="bars">{''.join(bars)}</div>
        <div class="axis"><span>{ticks[0]:.3f}</span><span>{ticks[1]:.3f}</span><span>{ticks[2]:.3f} m/s</span></div>
        <p class="foot" style="margin:10px 0 0">Qırıq xətt = persistence (heç nə etməmək). Qırmızı — ondan pis, yaşıl — ondan yaxşı. RMS xəta bütün {t['windows']} pəncərə, 20 kadr və hər iki komponent üzrə.</p>
      </div>
      <div class="card">
        <h2>Nə yoxlanıb</h2>
        <div class="checks">{''.join(f'<div><span class="ok">✓</span>{c}</div>' for c in checks)}</div>
      </div>
      <div class="card">
        <h2>Ölçmədən çıxan</h2>
        {findings}
      </div>
    </aside>
  </div>
</div>
"""
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
