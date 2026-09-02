# pkmngpt — Plan for the Damage Calculator & DQN Battler

This document plans two new components and the groundwork to do first. Read the
"Pre-work" section before starting either — several items unblock both, and a few
decisions change the plans substantially.

---

## Where the project stands (so the plans build on reality)

Working today:
- **Text-to-SQL** over a local PokéAPI SQLite DB (~95% on complex factual questions).
- **RAG** over Bulbapedia (creatures + concepts + lore) and Smogon analyses, in Chroma,
  with `doc_type` / `generation` / `tier` / `pokemon_name` metadata filters.
- **Smogon set store** — a keyed JSON of competitive movesets (`gen/tier/pokemon/setname`),
  handling the "value or list-of-alternatives" shape for every field.
- **Router** — classifies a question into SQL / Pokémon-RAG / concept-RAG / API-tool /
  smogon-sets / smogon-analysis, extracts pokemon_name/gen/tier with defaults and a
  fuzzy-matching fallback, then gathers evidence and synthesizes.

The two new components are a **different kind of work** from everything above. Everything
so far is *information retrieval* (answer a question from stored knowledge). The damage
calculator is *computation*. The DQN battler is *reinforcement learning*. Neither is a
question-answering feature, and the DQN especially is nearly a separate project.

---

## PART 0 — PRE-WORK (do before either component)

These are worth doing first because they either unblock both components or de-risk them.

### 0.1 — Pin down the damage calc's role in the wider system (decision, not code)
The calculator is the bridge: the router's synthesis layer can call it, *and* the DQN
needs it as its damage model. Decide now: is the calculator a **standalone tool the router
invokes** ("how much does X's move do to Y"), a **library the DQN imports**, or **both**?
It should be both, which means: build it as a clean, importable module with a simple
function signature — not buried inside the router. This decision shapes its API.

### 0.2 — Decide build-vs-buy for the damage formula (the single biggest decision)
The base damage formula is easy; the edge cases (abilities, items, weather, terrain,
screens, crits, multi-hit, the 16-roll spread, generation differences) are a swamp.
- **Buy:** `@smogon/calc` (JavaScript) is the exact library behind Smogon's calculator —
  battle-tested, handles every gen and interaction. Bridge from Python via a Node
  subprocess or a thin local service.
- **Build:** hand-roll the formula. Fine only if you scope hard (e.g. "Gen 9, level 100,
  the common modifiers") and accept being wrong on edge cases.
**Recommendation:** buy (`@smogon/calc`) if you want correctness across gens; build only a
scoped version if you want to stay pure-Python and accept limits. This decision determines
90% of the calculator plan — pick it before writing code.

### 0.3 — Reuse the Smogon set store as the calc's input format
The set store already holds movesets in a structured shape (moves, item, ability, EVs,
nature, Tera). That's most of what a damage calc needs for a "realistic set" calculation.
Confirm the set store exposes a clean accessor (`get_sets`) the calc can call to pull a
standard set when the user doesn't specify one. This is done — just verify the shape maps
to what the calc consumes.

### 0.4 — Base stats vs. battle stats (a correctness trap to internalize now)
PokéAPI gives **base stats**. A damage calc needs **actual in-battle stats**, computed from
base + level + IVs + EVs + nature. If you buy `@smogon/calc` it does this internally from a
set; if you build, you must implement the stat formula. Forgetting `base ≠ actual` is the
classic silent bug. Write and unit-test the stat formula early if building.

### 0.5 — A tiny, verifiable test set for the calculator
Before building, collect ~15–20 known damage results (from Smogon's public calculator:
"252+ Atk Choice Band X Move vs. Y" → a specific damage range). These are your ground
truth. The calc's eval is exact-match on the damage range — deterministic, like your SQL
result-comparison metric. Build the test set first so you can measure from day one.

### 0.6 — Confirm the GPU/compute story for the DQN (decision, not code)
The DQN needs many thousands of self-play battles. Decide the training environment now:
local CPU is too slow for meaningful RL; you'll want the school GPU. Confirm access,
queue/time limits, and whether jobs must be scripted+resumable (they will). This gates the
whole DQN timeline — resolve it before building, not mid-training.

### 0.7 — Decide the DQN's scope (the decision that most determines DQN size)
"A Pokémon battler" ranges from a student project to a thesis. Pick one, explicitly:
- **Narrow (recommended start):** a *fixed format and fixed teams* — e.g. Gen 1 OU with a
  small set of preset teams, or even 1v1 with a fixed roster. Small state space, tractable,
  "done" is definable.
- **Broad:** general team battling across a format. Much larger state space, far longer
  training, fuzzy "done."
The plan below assumes you start narrow and expand. Choosing broad first is the most common
way RL projects stall.

---

## PART 1 — THE DAMAGE CALCULATOR

A deterministic computation module: given attacker, defender, move, and conditions,
return the damage (a range, because of the random roll).

### 1.1 — Architecture
It is a **composite tool**: it needs inputs from multiple sources before it can compute.
Flow: extract/gather the battle scenario → resolve full sets (from the Smogon set store or
user-specified) → fetch any missing data (types, base stats — from PokéAPI) → compute →
format the result as a range with a KO verdict.

Build it as a standalone importable module (`damage_calc.py`) with a clean entry point:
```
calc_damage(attacker_set, defender_set, move, field_conditions) -> DamageResult
```
where `DamageResult` carries the min–max damage, the percent range, and the KO verdict
("guaranteed 2HKO", "possible OHKO", etc.). Both the router and the DQN import this.

### 1.2 — Phased build
**Phase A — the engine (buy or build, per 0.2).**
- *Buy path:* stand up the `@smogon/calc` bridge. Python builds the input JSON (attacker,
  defender, move, field), a Node subprocess runs the calc, returns the result JSON. Get one
  known calculation matching Smogon's site before anything else — that proves the bridge.
- *Build path:* implement the stat formula (0.4), then the damage formula with STAB, type
  effectiveness (you have `type_efficacy` in SQL), and the common modifiers (weather, item,
  ability, crit, the 16-roll spread). Scope to one generation. Unit-test against 0.5's set.

**Phase B — input resolution.**
Wire the calc to your existing data: pull base stats/types from PokéAPI, pull standard sets
from the Smogon set store when the user doesn't specify a spread. Handle the "value or
list-of-alternatives" set shape (you already solved this in `format_sets`). Decide defaults
(level 100, common spread) for unspecified fields.

**Phase C — natural-language extraction (the LLM part).**
"How much does Choice Band Garchomp's Earthquake do to Corviknight in sand?" → a structured
scenario. This is a **parse task**, exactly like your SQL/router extraction — a DSPy
signature: scenario text → {attacker, defender, move, item, ability, EVs, nature, field}.
Reuse the discipline that worked: state-only extraction (report what's said, calc layer
applies defaults), an alias map (Specs=Choice Specs, sun=Harsh Sunlight), and per-field
accuracy measured against a hand-labeled set. This is the piece most likely to need
few-shot demos.

**Phase D — router integration.**
Add a `needs_damage_calc` branch to the router (you know this pattern). It routes damage
questions to the extraction → calc → format pipeline, and the result becomes evidence the
synthesizer phrases. Add damage questions to the routing eval.

### 1.3 — Output must be a range, not a number
The 16-value random roll means damage is always a spread. Present it the way Smogon does:
`163–192 (44.9–52.9%) — guaranteed 2HKO`. A single number is wrong by construction.

### 1.4 — Eval
Exact-match the computed damage range against 0.5's known results (deterministic, cheap).
Separately, measure **extraction accuracy** (did it parse the scenario correctly) — that's
where the errors hide, since the calc itself is correct if you bought it.

### 1.5 — Rough sequencing
Engine (A) → input resolution (B) → extraction (C) → router integration (D). A and B are
the foundation; C and D make it usable through the assistant. The DQN only needs A and B.

---

## PART 2 — THE DQN SELF-PLAY BATTLER

A reinforcement-learning agent that learns to battle by playing itself. This is a separate
project in skill set (RL, not retrieval) and compute profile (GPU, long training).

### 2.1 — The foundation: a battle simulator (non-negotiable)
RL learns from playing *complete battles* at high speed — thousands to millions. You cannot
generate that from PokéAPI calls or the damage calc alone; you need a full battle engine
(turn order, switching, status, hazards, all the rules).
- **Do not write your own simulator.** Use **Pokémon Showdown's engine**, specifically the
  **`poke-env`** Python library, which wraps Showdown into a Gym-style RL environment
  (`reset()`, `step(action)`, a running Showdown server). This is ~80% of the "getting
  started" effort and it's plumbing, not ML.
- Your damage calculator (Part 1) is *not* the simulator — the simulator computes full
  battle outcomes. The calc is useful for the agent's *reasoning about a move*, but the
  environment is `poke-env`/Showdown.

### 2.2 — State and action representation (the real design work)
- **Action space** — small and discrete, which is why DQN fits: ~4 moves + ~5 switches ≈ a
  9-action space per turn. DQN is good at exactly this.
- **State encoding** — the hard part. The battle state (your active Pokémon's HP/stats/
  status/boosts, your bench, the opponent's revealed team, field/hazards) must become a
  fixed-length numeric vector. Get this wrong and no amount of training helps — it's the RL
  analogue of the chunking problem: the unglamorous representation decision that sets the
  ceiling.
- **Partial observability** — you don't see the opponent's full team/sets until revealed, so
  this is formally a POMDP, not a clean MDP. Vanilla DQN assumes full observability; it
  still works (people train Showdown agents this way) but it's a known ceiling.

### 2.3 — Self-play and reward
- **Reward** — naturally sparse: +1 win, −1 loss, 0 otherwise. Works but learns slowly. Most
  add shaping (small rewards for chip damage, KOs), with the caveat that bad shaping teaches
  the agent to farm the proxy instead of winning.
- **Self-play trap** — training only against the *current* self can collapse into cycles or
  overfit to its own quirks. Fix: an **opponent pool** — snapshot past versions and sample
  opponents from history. Start against Showdown's built-in heuristic bots for a sane
  baseline before pure self-play.
- **"Recursive learning"** (your phrasing) = self-play: the agent improves by playing
  successively stronger versions of itself. The opponent pool is what makes that stable.

### 2.4 — The DQN machinery (the standard, well-trodden part)
Q-network, replay buffer, target network, ε-greedy exploration; Double/Dueling DQN as
sensible defaults. Use **Stable-Baselines3** or **CleanRL** rather than hand-rolling the
algorithm — pair one of those with `poke-env`. Your novel work is the environment wrapper,
the state encoding, and the reward — not the DQN itself.

### 2.5 — Phased build
1. Stand up `poke-env` + a local Showdown server; get the RL loop running against built-in
   bots with a trivial reward. (Plumbing — prove the loop.)
2. Design the state encoding and action space; validate the agent *perceives* the battle
   (sanity-check the vector on known states).
3. Reward design + self-play with an opponent pool; training infrastructure
   (checkpoint/resume across GPU jobs — see 0.6).
4. Train to a competent agent (many hours–days of wall-clock; a few runs minimum).

### 2.6 — Where the damage calc connects
The agent may use the damage calc to evaluate "if I use this move, expected damage" as part
of its state features or action evaluation. That's the one genuine reuse between the two
components — build the calc's `calc_damage` clean and importable (0.1) so the agent can call
it. Otherwise the two projects are independent.

### 2.7 — Honest expectations
- Compute: RL is long and sample-hungry — the bottleneck is often *simulator throughput*
  (battles/sec), not the GPU. Your 24h job limit becomes relevant; plan checkpoint/resume.
- Iteration: reward shaping and state encoding are tuned by *watching the agent play badly*
  and adjusting — many runs, not one. "Works first try" is nearly a contradiction in RL.
- Scope (0.7) dominates everything: a fixed-team single-format agent is tractable; a general
  battler is months.

---

## Recommended overall order

1. **Pre-work 0.1–0.5** (calc decisions + test set) — cheap, unblocks the calc.
2. **Damage calculator Part 1, Phases A–B** — the engine + input resolution. This alone is
   useful (router can call it) *and* it's the DQN's damage model, so it's shared value.
3. **Damage calculator Phases C–D** — extraction + router integration, if you want it
   answerable through the assistant.
4. **Pre-work 0.6–0.7** (GPU + DQN scope decisions) — before touching RL.
5. **DQN Part 2** — as its own project, on the GPU, starting narrow (fixed team/format).

Do the calculator first: it's smaller, deterministic, immediately useful through your
existing router, and it's a dependency-free prerequisite the DQN benefits from. The DQN is
the larger, open-ended effort — firewall it as project two and start it narrow.

## The single most important caution
The DQN is a different discipline and a different time-scale from everything you've built.
Don't let its open-endedness stall the finished, working assistant. Ship the calculator
(bounded, deterministic, integrates with what exists), then start the DQN deliberately with
a narrow scope and realistic RL expectations — it's weeks-to-months, iteration-heavy, and
"done" is whatever you define it to be.
