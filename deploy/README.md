# Deploying this

Everything here is deployed. This file records what it took, including the two
things I got wrong on the way, because the wrong version is instructive.

| | |
|---|---|
| Replay (a finished run) | https://dme-coordination-replay.jatingoyal.workers.dev |
| **The simulator, live** | **https://dme-simulator.jatingoyal.workers.dev** |
| Spine probe | https://dme-spine-probe.jatingoyal.workers.dev |

## What I said first, and why it was wrong

I wrote that hosting the simulator on Workers meant porting `policy.py` and
`engine.py` to TypeScript, and that Workers could not carry a run that takes
minutes. Both were wrong:

- **Python Workers run Pyodide**, and the deterministic half of this system is
  pure standard-library Python. It runs there unmodified — the spine probe
  proves it, returning identical clock arithmetic, the same `Wait until Thu
  08:00` at 02:14, the same qualification verdicts, and the same behaviour from
  all three model-output guards.
- **There is no wall-clock cap** while the client stays connected. The docs are
  explicit: *"There is no set time limit on individual subrequests."*

## What it actually took

**One refactor, at the edges.** Workers have no sockets: outbound HTTP is
`fetch`, and `fetch` is async, so a synchronous call path cannot await it. The
chain `engine.step → caller._run → llm.say` became `async`. **`policy.py` did
not change at all**, because it performs no I/O — the dividend of having put
every decision in a pure module. A test now enforces that: `policy` may contain
no coroutines and may not import anything that talks to a network.

**Three smaller things:**

- **The `openai` SDK is gone.** It wants sockets. The request is one POST with a
  JSON body, so `dme/llm.py` makes it by hand and picks a transport at import —
  `fetch` in a Worker, `urllib` on a machine. `requirements.txt` is now empty:
  the whole system runs on the standard library. (`urllib` also has to announce
  a User-Agent, or Groq's edge answers 403.)
- **`tzdata` as a dependency.** Pyodide ships no time zone database, so
  `ZoneInfo("America/Chicago")` raises until you declare it.
- **No filesystem.** `scripts/build_worker.py` inlines `data/*` into a generated
  module, and `dme/loader.py` falls through to it.

**No threads.** `threading` imports on Pyodide and does nothing. The streaming
used a worker thread and a queue; now the engine's hooks buffer and the driver
flushes after each `engine.step()`. That is simpler than what it replaced, and
it is the same code locally and on Workers — both stream 42 messages for the
same seeded case and reach the same outcome.

## Deploying it yourself

```sh
python scripts/build_worker.py          # sync dme/, inline data/, copy the app
cd deploy/py && uv run pywrangler deploy
```

Then set the secrets — they are never in the bundle:

```sh
cd deploy/py && set -a && . ../../.env && set +a && \
  echo -n "$GROQ_API_KEY"   | npx wrangler secret put GROQ_API_KEY && \
  echo -n "$SARVAM_API_KEY" | npx wrangler secret put SARVAM_API_KEY
```

## The one real limit: subrequests

**Workers Free allows 50 subrequests per invocation; Paid allows 10,000** (the
old 1,000 cap was lifted in February 2026). Every model call is a subrequest, and
one supplier call costs roughly ten — a conversation is two model calls per turn
plus an extraction.

Measured on the deployed Worker:

| directory | outcome |
|---|---|
| 3 suppliers | resolves end to end — 54 messages, 6 phone calls, 0.46 simulated days |
| 4 suppliers | gets as far as booking the qualified supplier, then runs out |
| 12 (the brief's full directory) | ~145 model calls; needs a paid plan or the local runner |

So the hosted copy trims its pre-filled directory to three and says so in the
form. Running out is reported as what it is — a plan setting, with the advice to
use fewer suppliers or run locally — rather than as a `JsException` from three
layers down. `SubrequestLimit` is its own type, and two tests keep it distinct
from a provider error.

The full case runs locally with no limit at all:

```sh
python3 scripts/simulate.py
```
