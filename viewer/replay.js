/* The renderer, shared by the finished-run replay and the live simulator.
 *
 * It draws whatever is in RUN and knows nothing about where that came from --
 * a JSON blob baked into a file, or events arriving over a stream as a case is
 * actually being worked. One renderer means the demo and the artefact cannot
 * disagree about what happened.
 */
const Viewer = (() => {
  const $ = (s) => document.querySelector(s);
  const el = (t, c, x) => {
    const n = document.createElement(t);
    if (c) n.className = c;
    if (x !== undefined) n.textContent = x;
    return n;
  };

  const GATES = ["accepts_new_medicare_patients", "stocks_k0001", "accepts_assignment", "serves_patient_area"];
  const SHORT = {
    accepts_new_medicare_patients: "new pts", stocks_k0001: "in stock",
    accepts_assignment: "assignment", serves_patient_area: "their area",
  };
  const ORDER_EVENTS = new Set(["ClinicCalled", "OrderRequested", "OrderPromised", "OrderReceived",
    "OrderRefused", "CommitmentMade", "CommitmentBroken", "OrderSentToSupplier"]);
  const PATIENT_EVENTS = new Set(["PatientContacted", "ConsentRecorded"]);

  let RUN = null, idx = 0, timer = null, scale = "time", follow = false, onVoice = null,
      voiceAvailable = () => true;

  const supById = () => Object.fromEntries((RUN.suppliers || []).map(s => [s.id, s]));
  const fmt = (iso) => new Date(iso).toLocaleString("en-GB",
    { weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  const day = (iso) => new Date(iso).toLocaleDateString("en-GB",
    { weekday: "short", day: "2-digit", month: "short" });

  function pctAt(i) {
    const steps = RUN.steps;
    if (steps.length < 2) return 0;
    if (scale === "steps") return (i / (steps.length - 1)) * 100;
    const t0 = new Date(steps[0].at).getTime();
    const t1 = new Date(steps[steps.length - 1].at).getTime();
    return ((new Date(steps[i].at).getTime() - t0) / Math.max(1, t1 - t0)) * 100;
  }

  function laneOf(s) {
    if (ORDER_EVENTS.has(s.type) || s.type === "CallClinic" || s.type === "FaxClinic") return "order";
    if (PATIENT_EVENTS.has(s.type) || s.type === "ContactPatient") return "patient";
    return "supplier";
  }

  function label(s) {
    if (s.kind === "event") return s.line;
    const by = supById();
    const who = s.supplier_id && by[s.supplier_id] ? by[s.supplier_id].name : "";
    switch (s.type) {
      case "CallSupplier": return "call " + who;
      case "CallClinic": return "call the clinic";
      case "FaxClinic": return "fax the clinic";
      case "ContactPatient": return "contact the patient about " + s.topic;
      case "SendOrderToSupplier": return "send the written order to " + who;
      case "ScheduleDelivery": return "book delivery with " + who;
      case "Escalate": return "escalate: " + (s.reason || "").replace(/_/g, " ");
      case "CloseCase": return "close the case";
      case "Wait": return s.until ? "wait until " + fmt(s.until) : "wait";
      default: return s.type;
    }
  }

  /* ---- tiles ----------------------------------------------------------- */
  function tiles() {
    const box = $("#tiles"); box.innerHTML = "";
    const add = (k, v, n, hero) => {
      const t = el("div", "tile" + (hero ? " hero" : ""));
      t.append(el("div", "k", k), el("div", "v", v));
      if (n) t.append(el("div", "n", n));
      box.append(t);
    };
    const o = RUN.outcome, u = RUN.usage || {};
    if (!o) {
      add("outcome", "Working the case…", (RUN.steps.length || 0) + " decisions so far", true);
    } else if (o.resolved) {
      add("outcome", "Resolved, no human", "delivering " + day(o.delivery_at), true);
    } else if (o.escalation) {
      add("outcome", "Escalated to a human", o.escalation.replace(/_/g, " "), true);
    } else {
      add("outcome", "Stalled", o.stall_note || "", true);
    }
    const last = RUN.steps.length ? RUN.steps[RUN.steps.length - 1] : null;
    const days = o ? o.elapsed_days
      : (last ? (new Date(last.at) - new Date(RUN.steps[0].at)) / 86400000 : 0);
    add("simulated elapsed", days.toFixed(1) + " d", "from " + (RUN.case.opened_at
      ? new Date(RUN.case.opened_at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })
      : "") + ", everyone asleep");
    add("phone calls", o ? o.calls_placed : Object.keys(RUN.calls || {}).length,
        (RUN.suppliers || []).length + " suppliers in the directory");
    add("model calls", u.calls || 0,
        u.input_tokens ? ((u.input_tokens / 1000).toFixed(0) + "k in / " + (u.output_tokens / 1000).toFixed(0) + "k out") : "—");
    const vetoed = o ? o.vetoed : Object.values(RUN.calls || {}).reduce((n, c) => n + ((c.dropped || []).length), 0);
    add("answers vetoed", vetoed, vetoed ? "quoted but not in the call" : "every fact traced to a quote");
    const blocked = o ? (o.blocked || 0)
      : Object.values(RUN.calls || {}).reduce((n, c) => n + ((c.blocked || []).length), 0);
    add("identifiers refused", blocked, blocked ? "our agent tried to invent one" : "nothing invented on a call");
  }

  /* ---- timeline -------------------------------------------------------- */
  function lanes() {
    const host = $("#lanes"); host.innerHTML = "";
    const byLane = {};
    for (const [key, name] of [["supplier", "supplier search"], ["order", "written order"], ["patient", "patient"]]) {
      const nm = el("div", "lane-name");
      const sw = el("i", "swatch"); sw.style.background = "var(--" + key + ")";
      nm.append(sw, document.createTextNode(name));
      const ln = el("div", "lane");
      host.append(nm, ln);
      byLane[key] = ln;
    }
    RUN.steps.forEach((s, i) => {
      const m = el("div", "mark" + (s.call_id ? " big" : ""));
      m.style.left = pctAt(i) + "%";
      const colour = "var(--" + laneOf(s) + ")";
      if (s.kind === "decision") {
        m.style.background = "var(--surface-1)";
        m.style.boxShadow = "inset 0 0 0 2px " + colour;
      } else m.style.background = colour;
      m.dataset.i = i;
      m.addEventListener("mouseenter", (e) => tip(e, fmt(s.at) + " — " + (s.line || label(s))));
      m.addEventListener("mouseleave", hideTip);
      m.addEventListener("click", () => { goto(i); if (s.call_id) showCall(s.call_id); });
      byLane[laneOf(s)].append(m);
    });
    const ph = el("div", "playhead"); ph.id = "ph"; host.append(ph);

    const ticks = $("#ticks"); ticks.innerHTML = "";
    const seen = new Set(); let lastLeft = -99;
    RUN.steps.forEach((s, i) => {
      const d = day(s.at);
      if (seen.has(d)) return;
      const left = pctAt(i);
      if (left - lastLeft < 7) return;
      seen.add(d); lastLeft = left;
      const t = el("div", "tick", d); t.style.left = left + "%"; ticks.append(t);
    });
  }

  /* ---- feed ------------------------------------------------------------ */
  function feed() {
    const host = $("#feed"); host.innerHTML = "";
    const legend = el("div", "actors");
    [["policy", "decided by code"], ["llm", "a model spoke or read"],
     ["engine", "performed / kept time"], ["world", "happened to us"]].forEach(([k, what]) => {
      const row = el("span");
      row.append(el("i", "by by-" + k, k), document.createTextNode(what));
      legend.append(row);
    });
    host.append(legend);
    RUN.steps.forEach((s, i) => {
      const r = el("div", "row " + (s.kind === "decision" ? "decision" : "event"));
      r.dataset.i = i;
      r.append(el("span", "t", fmt(s.at).slice(0, 12)));
      const b = el("div", "b");
      if (s.kind === "decision") {
        b.append(el("span", "arrow", "→ "), document.createTextNode(label(s)));
        if (s.why) b.append(el("div", "why", s.why));
      } else {
        b.append(document.createTextNode(s.line));
        if (s.attribution) b.append(el("div", "why-line", s.attribution));
      }
      r.append(b);
      if (s.by) {
        const badge = el("span", "by by-" + s.by, s.by);
        badge.title = s.attribution || "";
        r.append(badge);
      }
      if (s.call_id) {
        r.classList.add("clickable");
        r.addEventListener("click", () => { goto(i); showCall(s.call_id); });
      }
      host.append(r);
    });
  }

  /* ---- board ----------------------------------------------------------- */
  function board() {
    const f = RUN.frames[idx];
    const host = $("#board"), tr = $("#tracks");
    host.innerHTML = ""; tr.innerHTML = "";
    if (!f) return;
    const by = supById();

    const o = f.order;
    const ot = el("div", "track"); ot.style.borderColor = "var(--order)";
    const oh = el("div", "h");
    const os = el("i", "swatch"); os.style.background = "var(--order)";
    oh.append(os, document.createTextNode("Written order"));
    const op = el("span", "pill" + (o.status === "received" ? " on" : ""), o.status.replace(/_/g, " "));
    op.style.marginLeft = "auto"; oh.append(op); ot.append(oh);
    ot.append(el("div", "d", (o.attempts ? o.attempts + " call(s) to the clinic" : "not yet contacted")
      + (o.faxed ? " · faxed" : "") + (o.coded_as ? " · coded " + o.coded_as : "")
      + (o.sent_to && by[o.sent_to] ? " · sent to " + by[o.sent_to].name : "")));
    tr.append(ot);

    const p = f.patient;
    const pt = el("div", "track"); pt.style.borderColor = "var(--patient)";
    const ph2 = el("div", "h");
    const ps = el("i", "swatch"); ps.style.background = "var(--patient)";
    ph2.append(ps, document.createTextNode(RUN.case.patient.name));
    const pp = el("span", "pill" + (p.consent === "yes" ? " on" : ""), "consent: " + p.consent);
    pp.style.marginLeft = "auto"; ph2.append(pp); pt.append(ph2);
    pt.append(el("div", "d", p.cost_explained
      ? "rang them: 80/20 plus deductible explained, no figure quoted"
      : "not yet contacted"));
    if (p.cost_explained) {
      // The patient is rung like everyone else now, so her call is a transcript.
      const call = Object.keys(RUN.calls || {}).find(
        id => (RUN.calls[id].with || "") === RUN.case.patient.name);
      if (call) {
        const link = el("div", "d");
        const a = el("span", "gate days", "listen to the call");
        a.style.cursor = "pointer";
        a.addEventListener("click", () => showCall(call));
        link.append(a); pt.append(link);
      }
    }
    tr.append(pt);

    (f.tasks || []).forEach(t => {
      const card = el("div", "track");
      card.style.borderColor = t.open ? "var(--serious)" : "var(--ink-3)";
      const h = el("div", "h");
      const sw = el("i", "swatch");
      sw.style.background = t.open ? "var(--serious)" : "var(--ink-3)";
      h.append(sw, document.createTextNode("A person was asked to " + t.kind.replace(/_/g, " ")));
      const pill = el("span", "pill" + (t.open ? "" : " on"), t.open ? "waiting" : "done");
      pill.style.marginLeft = "auto";
      h.append(pill);
      card.append(h);
      card.append(el("div", "d", t.blocks
        ? "the " + t.blocks + " track waits on this; everything else keeps running"
        : "nothing waits on this — the case carries on"));
      tr.append(card);
    });

    (RUN.suppliers || []).forEach(s => {
      const st = f.suppliers[s.id];
      if (!st) return;
      const card = el("div", "sup");
      const nm = el("div", "nm");
      nm.append(el("span", null, s.name));
      const colour = { qualified: "var(--good)", disqualified: "var(--ink-3)",
        unreachable: "var(--serious)", in_progress: "var(--warning)", uncontacted: "var(--ink-3)" }[st.status];
      const stx = el("span", "st", { qualified: "✓ qualified", disqualified: "✗ ruled out",
        unreachable: "— unreachable", in_progress: "? partial", uncontacted: "· not called" }[st.status]);
      stx.style.color = colour; nm.append(stx); card.append(nm);
      const g = el("div", "gates");
      GATES.forEach(k => {
        const fact = st.facts[k];
        const v = fact ? fact.v : "unknown";
        const chip = el("span", "gate " + (v === "yes" ? "yes" : v === "no" ? "no" : ""),
          (v === "yes" ? "✓ " : v === "no" ? "✗ " : "? ") + SHORT[k]);
        if (fact) {
          chip.title = "from " + fact.src;
          chip.addEventListener("click", () => showCall(fact.src.replace("call:", "")));
        }
        g.append(chip);
      });
      if (st.facts.earliest_delivery_days) {
        const d = el("span", "gate days", st.facts.earliest_delivery_days.v + "d");
        d.addEventListener("click", () => showCall(st.facts.earliest_delivery_days.src.replace("call:", "")));
        g.append(d);
      }
      card.append(g);
      if (st.out) card.append(el("div", "out", st.out));
      if (f.chosen === s.id) card.style.borderColor = "var(--good)";
      host.append(card);
    });
  }

  /* ---- call panel ------------------------------------------------------ */
  function showCall(id) {
    const c = (RUN.calls || {})[id];
    const host = $("#panel");
    if (!c) { host.className = "empty"; host.textContent = "No transcript for " + id + " yet."; return; }
    $("#paneltitle").textContent = "Call " + id + " — " + c.with;
    host.className = ""; host.innerHTML = "";
    c.lines.forEach((l, i) => {
      const r = el("div", "line" + (l.who === "agent" ? " us" : ""));
      const who = el("div", "who", l.who === "agent" ? "us" : "them");
      if (onVoice) {
        const b = el("button", "audio-btn", "▸");
        const on = voiceAvailable();
        b.title = on
          ? "hear this line at 8 kHz — the bandwidth a phone call has"
          : "no SARVAM_API_KEY on this server";
        if (!on) b.classList.add("mute");
        b.addEventListener("click", () => onVoice(b, l.who, c.with, l.text));
        who.append(b);
      }
      r.append(who, el("div", "tx", l.text));
      host.append(r);
    });
    const learned = el("div", "learned");
    const facts = c.facts || {}, keys = Object.keys(facts);
    learned.append(el("b", null, "Extracted: "));
    learned.append(document.createTextNode(keys.length
      ? keys.map(k => (SHORT[k] || k) + " = " + facts[k]).join(", ")
      : "nothing — every field stayed unknown"));
    if ((c.dropped || []).length) {
      learned.append(el("div", "veto", "Vetoed as ungrounded: " + c.dropped.join(", ")
        + " — the model answered, but could not quote them saying it."));
    }
    if ((c.blocked || []).length) {
      learned.append(el("div", "veto", "Guard fired: our agent stated a " + c.blocked.join(", ")
        + " it was never given. The turn was replaced before it was spoken."));
    }
    if (c.safety_stop) {
      learned.append(el("div", "veto",
        "Safety stop: they made a patient identifier a condition of helping."));
    }
    host.append(learned);

    // What this call actually caused. The transcript is only half the story --
    // the interesting half is which component did what with it afterwards.
    const at = RUN.steps.findIndex(s => s.call_id === id && s.kind === "event");
    if (at >= 0) {
      const after = [];
      for (let i = at; i < RUN.steps.length && after.length < 6; i++) {
        const s = RUN.steps[i];
        if (i > at && s.call_id && s.call_id !== id) break;   // the next call starts
        if (s.kind === "event") after.push(s);
      }
      if (after.length) {
        const led = el("div", "led");
        led.append(el("b", null, "What this call led to"));
        after.forEach(s => {
          const row = el("div", "step");
          const badge = el("span", "by by-" + s.by, s.by);
          badge.title = s.attribution || "";
          row.append(badge, el("span", "tx", s.line));
          led.append(row);
          if (s.attribution) led.append(el("div", "why-line", s.attribution));
        });
        host.append(led);
      }
    }
  }

  function showMessage(i) {
    const m = (RUN.messages || [])[i];
    if (!m) return;
    $("#paneltitle").textContent = "Message to " + RUN.case.patient.name + " — " + m.topic;
    const host = $("#panel"); host.className = ""; host.innerHTML = "";
    host.append(el("div", "msg", m.text));
    host.append(el("div", "learned", m.from_model
      ? "Drafted by the model, and it passed the content check: shares named, no figure invented."
      : "The model's draft failed the content check, so the fixed template was sent instead."));
  }

  /* ---- playback -------------------------------------------------------- */
  function goto(i) {
    idx = Math.max(0, Math.min(RUN.steps.length - 1, i));
    const scrub = $("#scrub");
    scrub.max = Math.max(0, RUN.steps.length - 1);
    scrub.value = idx;
    render();
  }

  function render() {
    if (!RUN.steps.length) return;
    const s = RUN.steps[idx];
    $("#stamp").textContent = fmt(s.at);
    const ph = $("#ph");
    if (ph) ph.style.left = "calc(108px + (100% - 108px) * " + (pctAt(idx) / 100) + ")";
    document.querySelectorAll(".mark").forEach(m => m.classList.toggle("future", +m.dataset.i > idx));
    document.querySelectorAll(".feed .row").forEach(r => {
      const i = +r.dataset.i;
      r.classList.toggle("dim", i > idx);
      r.classList.toggle("now", i === idx);
    });
    const now = $(".feed .row.now"), box = $("#feed");
    if (now && box) box.scrollTop = Math.max(0, now.offsetTop - box.clientHeight * 0.6);
    board();
  }

  function play() {
    const btn = $("#play");
    if (timer) { clearInterval(timer); timer = null; btn.textContent = "▶ play"; return; }
    if (idx >= RUN.steps.length - 1) idx = -1;
    btn.textContent = "❚❚ pause";
    timer = setInterval(() => {
      if (idx >= RUN.steps.length - 1) { clearInterval(timer); timer = null; btn.textContent = "▶ play"; return; }
      goto(idx + 1);
    }, 320);
  }

  /* ---- tooltip --------------------------------------------------------- */
  function tip(e, text) {
    const t = $("#tip"); if (!t) return;
    t.textContent = text; t.style.opacity = 1;
    t.style.left = Math.min(window.innerWidth - 340, e.clientX + 12) + "px";
    t.style.top = (e.clientY + 14) + "px";
  }
  function hideTip() { const t = $("#tip"); if (t) t.style.opacity = 0; }

  /* ---- api ------------------------------------------------------------- */
  function header() {
    const c = RUN.case;
    $("#caseline").textContent = c.patient.name + (c.patient.age ? ", " + c.patient.age : "")
      + " — " + c.equipment + " (" + c.hcpcs + "), " + c.pcp;
    if ($("#worldline")) $("#worldline").textContent = c.world || "";
    const box = $("#assumptions");
    if (box) {
      box.innerHTML = "";
      (c.assumptions || []).forEach(a => {
        const [head, ...rest] = a.split(":");
        const d = el("div");
        d.append(el("b", null, "assumption · " + head + ": "));
        d.append(document.createTextNode(rest.join(":").trim()));
        box.append(d);
      });
    }
  }

  function refresh(keepPosition) {
    header(); tiles(); lanes(); feed();
    if (follow || !keepPosition) goto(RUN.steps.length - 1); else render();
    const packet = $("#packetcard");
    if (packet && RUN.outcome && RUN.outcome.packet) {
      packet.hidden = false;
      $("#packet").textContent = JSON.stringify(RUN.outcome.packet, null, 2);
    }
  }

  function init(run, opts) {
    RUN = run;
    opts = opts || {};
    follow = !!opts.follow;
    onVoice = opts.onVoice || null;
    if (opts.voiceAvailable) voiceAvailable = opts.voiceAvailable;
    $("#scrub").addEventListener("input", (e) => { follow = false; goto(+e.target.value); });
    $("#play").addEventListener("click", play);
    const sc = $("#scale");
    if (sc) sc.addEventListener("click", () => {
      scale = scale === "time" ? "steps" : "time";
      sc.textContent = "scale: " + (scale === "time" ? "elapsed time" : "one slot per step");
      lanes(); render();
    });
    const th = $("#theme");
    if (th) th.addEventListener("click", () => {
      const dark = document.documentElement.getAttribute("data-theme") === "dark";
      document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    });
    document.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (e.key === " ") { e.preventDefault(); play(); }
      if (e.key === "ArrowRight") { follow = false; goto(idx + 1); }
      if (e.key === "ArrowLeft") { follow = false; goto(idx - 1); }
    });
    window.addEventListener("resize", render);
    if (RUN.steps.length) refresh(false);
  }

  return { init, refresh, goto, showCall, showMessage, setFollow: (v) => { follow = v; } };
})();
