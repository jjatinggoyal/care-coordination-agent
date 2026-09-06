# DME back-end coordination

Works a patient's durable-medical-equipment case with nobody awake to supervise
it — calls suppliers to find one that can actually serve them, chases the
physician's office for the written order, rings the patient to explain what
they'll owe, books the delivery, and asks a human only when it genuinely has to.

---

## Run the demo

**Quickest: open the hosted simulator — nothing to install, no key needed.**

### → **https://dme.jatingoyal.com**

Fill in a case and press **Run**; it works the case live, in front of you.

- Every `→` is the policy choosing, with its reasoning underneath.
- Each step is badged **policy / llm / engine / world** — who decided, and why.
- Click any call to read it. Press **▸** on a line to hear it at 8 kHz.
- Change anything: the patient, the equipment, the directory, when the case
  opens, who's on the other end of each phone.

It's pre-filled with **the brief's full twelve-supplier directory** and resolves
end to end in about two minutes. Change any of it; every persona is in the
dropdown.

**How it fits on a free plan.** Cloudflare allows 50 outbound requests per
*invocation*, and a whole case needs several hundred. So the browser drives the
loop: one request per `engine.step()`, each with its own budget, with the event
log riding along between them. That works because **the ledger already is the
state** — folding the same events rebuilds the same case, so both the case and
the simulated world are reconstructed from history on every request and nothing
is held server-side. A full run is ~29 requests carrying ~52 events.

There's also [a finished twelve-supplier
run](https://dme-replay.jatingoyal.com) you can scrub
through, if you'd rather read one than wait for one.

**Locally — the full case.** Nothing to install; it runs on the Python standard
library.

```sh
echo "GROQ_API_KEY=gsk_..." > .env      # add SARVAM_API_KEY too, for audio
python3 scripts/simulate.py             # → http://127.0.0.1:8800
```

Other ways in:

```sh
python3 scripts/run_case.py --transcripts   # the brief's case, in the terminal
python3 scripts/run_eval.py -n 6            # the sweep: six seeded worlds
python3 -m unittest discover -s tests       # 84 tests, no API key needed
```

That the tests need no API key is the whole architectural claim, demonstrated.

---

## What I built, and how I decided

- **Read the brief as a durable workflow, not a chatbot.** A case lives for days
  across three parties who are asleep or busy most of it. The only integration
  surface is a telephone. Nothing about that wants an agent loop at the centre.
- **So: build a world, then a policy engine, then put an LLM on top.** In three
  hours that split buys the most.
  - **A simulated world** — suppliers and clinic staff who don't pick up, don't
    know, say yes and mean nothing. Built *early*, not last: without it there's
    nothing to test against, and once it exists the eval falls out free. **The
    persona deck is the eval set.**
  - **A deterministic policy engine** — the operating procedure of a care
    advocate as pure functions. Had passing tests before a single API call.
  - **An LLM at the edges only** — to hold a phone call and to read a transcript.
- **Build order:** clock and ledger → policy → one leg deep (supplier
  qualification by phone) → the written-order chase → the join, the cost call,
  the handoff, the booking.

---

## Technology & architecture

**Stack:** Python, standard library only. `requirements.txt` is empty. **No
orchestration framework**.

**The pipeline:**

```
policy.decide(case, now) → Action     pure. no I/O, no model, no network
engine performs the Action            holds no opinions
the result becomes an Event           the only way state ever changes
reducer folds it into state           provenance attached here
```

- State is a fold of history; `ledger.replay()` rebuilds it exactly — which is
  what lets the hosted copy run a case across ~29 stateless HTTP requests, the
  browser carrying the log and the server holding nothing.
- The simulated world is stateless too: whether a phone is answered is a hash of
  `(seed, who, attempt)`, and everything else it needs it reads off the case.
- Nothing is scheduled. Retry times are *derived* from call history each step —
  no timer can drift or leak.
- **Ten typed actions** are the entire vocabulary between decision and effect.
- `RequestHumanTask` is the interesting one: asks a person for **one step** and
  suspends **one track**, so the supplier search keeps running while somebody
  posts a document. Escalating means the case is no longer ours; a human task
  means we need a hand.

**Where a model runs — two files, five call sites:**

| file | sites | job |
|---|---|---|
| `agents/caller.py` | 1 | one turn of our side of a phone call |
| `agents/extract.py` | 4 | supplier / clinic / booking / patient transcript → typed facts |

**Two jobs: hold a call, read a transcript.** Everything else is `policy.py` —
zero references to a model, imports nothing that reaches a network, tests
enforce both.

A sixth call site, `sim/world.py`, plays the humans. Scaffolding, not product:

```
182 model calls — caller=80, extractor=22, sim_supplier=67, sim_clinic=6, sim_patient=7
```

**102 of 182 are the system; 80 are the world pretending to be people.** In
production those 80 are humans on real phones.

**Six things keep the boundary honest:**

1. **No confidence scores.** A float invites a threshold; a threshold is where
   model judgement re-enters. `unknown` is a real state the policy handles.
2. **Evidence checked in Python** — every non-unknown answer must quote its
   justification or it's discarded back to `unknown`.
3. **Quotes match only the *other party's* turns** — earned, after the caller
   model wrote both halves of a conversation and the extractor believed the
   invented "yes"es.
4. **A promise is never an outcome** — every commitment carries a verification
   time, taken from the timeframe they gave, capped at three days.
5. **A guard on our agent's speech** — it can't state an identifier it was never
   given (it once read out `1234 West Maple Street`), or quote a dollar figure.
6. **Agreement without understanding is not consent.** A real call had the
   patient say "I don't know what you mean by the deductible", then "yes, go
   ahead". The extractor reported both; proceeding anyway would have been a
   choice, and the wrong one.

**Measured over six seeded worlds** (`samples/eval.txt`):

| | |
|---|---|
| resolved with no human | **4 / 6** |
| median simulated days · phone calls | 4.3 · 18 |
| escalations | `order_unobtainable`, `order_coding_mismatch` |

- Both escalations are **correct**, not failures.
- Calls per case went **up** over the build, 14 → 18, deliberately — asking how
  fast each supplier is stopped it booking a three-week delivery by accident.

---

## The cut list

**Product — things a real patient would notice missing**

- **No caregiver or second contact.** For a 72-year-old the daughter is often the
  actual coordinator. There's one phone number and it's the patient's.
- **English only.** Not a safe assumption for this population in Chicago.
- **Phone-only, in a workflow about mobility and often hearing.** The channel
  excludes some of the people it's for.
- **No status for anyone.** The patient can't ask "where is my wheelchair", and
  the care advocate can't see a queue of cases.
- **The human who picks up an escalation gets JSON**, not a screen.
- **Every case is equal.** Someone waiting three weeks doesn't outrank a routine
  one. No prioritisation, no SLA, no ageing.
- **We never confirm delivery.** Track shipping through shipping provider's
tracking number, and inform the same to the patient.

**Technical**

- **Real telephony.** `adapters/PhoneTransport` is the seam; the simulator
  satisfies it structurally.
- **True concurrency.** Tracks suspend independently; the engine still performs
  one action at a time.
- **A real voice pipeline.** Sarvam renders lines at 8 kHz — the bandwidth a
  phone call has — and that's all. No carrier, no STT, no turn-taking.
- **Any dollar figure.** No fee schedule, no deductible balance, so a guard
  blocks any amount. Deliberate, and the first thing typed coverage terms would
  undo — the guard becomes the check that verifies a real number.
- **Supplier ranking** — nothing to rank on yet. **Contradiction detection** —
  the ledger holds what's needed, the check isn't written.
- Persistence, auth, eligibility checks, prior auth, competitive bidding, ABNs.

---

## What's next

### One day

- **Send her something in writing** after the call — the shares, the supplier,
  the date. She asked for it unprompted. Highest value per hour in the list.
- **A caregiver field**, and ring them when the patient can't be reached.
- **Confirm delivery happened** before closing the case.
- *Technical:* true concurrency; a real transport behind `PhoneTransport`.

### Two weeks

- **The write path.** Every call produces a fact about a supplier — who picks up,
  who delivers, who wastes a week — and all of it is discarded when the case
  closes. Record it and the sparse 12-row directory becomes a ranked,
  self-maintaining asset. The fiftieth case shouldn't repeat the nine calls the
  first one made. It's also exactly the knowledge a care advocate carries in
  their head and can't hand over.
- **A console for the human who picks up an escalation** — the handoff packet as
  a screen, with the transcripts one click away.
- **Case ageing and prioritisation.** How long has this person been waiting.
- **Know what she actually owes.** Her plan is a string today — no deductible,
  no balance, no fee schedule — so the system can only quote percentages.
  Typed coverage terms would let it give her a real number.

### One month

- **The world talks back.** Today every fact is *pulled* by dialling out. The
  real world *pushes*: the supplier rings to confirm or cancel, the signed order
  arrives as a fax or an emailed PDF, a delivery confirmation comes by SMS — or
  nothing arrives at all. Needs an owned number with an inbound agent, a
  fax/email inbox with document matching, and **correlation** (an inbound call
  from a supplier serving four of our patients belongs to *which* case?). Plus a
  policy for unsolicited news: a supplier ringing to cancel should *reopen* a
  closed track.
- **Beyond wheelchairs - Configurable Workflows** — CPAP, hospital beds, oxygen. The policy generalises;
  the qualification gates and the paperwork don't.
- **Spanish, and a hearing-accessible channel.**
- **Operational dashboard** — days-to-delivery, calls-per-case, escalation rate
  by reason. The eval already computes all three; nobody can see them.
- **Reconcile what people tell us against what's actually true.** Every fact in
  this system is somebody's word on a phone, and nothing is checked against a
  source — a supplier quoting 30% when the plan says 20%, asking for a
  prescription the order on file already satisfies, or offering a date the
  item's rental rules don't allow. Plan terms, fee schedules, coverage criteria
  and the order itself are all real documents. Typed, they turn every claim into
  something checkable, and a discrepancy gets raised on the call instead of
  discovered on a bill.
