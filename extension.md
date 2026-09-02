# Extension: backend-specific "training"

pkmngpt's generation backend is a config flag (`LLM_BACKEND=ollama` or
`claude`, set in `.env`). The two backends have fundamentally different
options for improving generation quality over time, so they get different
treatment:

| Backend | Improvement path |
|---|---|
| `claude` | DSPy prompt optimization only (`BootstrapFewShot` / `MIPRO` against `eval/eval_set.json`) |
| `ollama` | DSPy prompt optimization **+** actual LoRA/QLoRA fine-tuning of the local model's weights |

This split exists because Anthropic doesn't expose a fine-tuning API for
Claude — DSPy optimization (picking/synthesizing better few-shot examples
and prompt phrasing) is the ceiling for that path. Ollama runs an
open-weight local model, so its weights are actually trainable.

Not implemented yet — this file documents the intended shape so the two
paths can be built out without re-deriving the plan.

---

## Path 1: `claude` backend — DSPy optimization only

Unchanged from the current plan already noted in `README.md`'s "Not yet
done" section:

1. Grow `eval/eval_set.json` into a real devset.
2. Run `dspy.BootstrapFewShot` (or `MIPRO`) against `PokemonRAG` using that
   devset and the `keyword`/`judge` metric.
3. The optimizer compiles a fixed, improved prompt/few-shot set. Every
   future call uses it at constant token cost — nothing grows per-query.
4. Re-compile only when the eval set grows or the metric plateaus, not on
   every run.

No further "training" beyond this exists for the Claude path. Retrieval
(`src/pokeapi.py`) is identical for both backends and isn't affected by
either optimization path.

---

## Path 2: `ollama` backend — DSPy optimization + fine-tuning

Same DSPy optimization step as above, **plus** an actual weight-level
fine-tune of the local model. This is a separate ML workflow from the RAG
codebase — it doesn't touch `src/llm.py`, `src/rag.py`, or `src/pokeapi.py`
at all. It produces a *new local model* that `OLLAMA_MODEL` in `.env`
points at afterward. Everything downstream of that config value works
unchanged.

### Why fine-tuning is viable here but not on Claude

Ollama serves open-weight models (e.g. `llama3.1`) from local GGUF files.
Because you hold the weights, you can retrain them. **LoRA/QLoRA**
(parameter-efficient fine-tuning) is the practical way to do this on
consumer hardware — instead of updating all of a model's billions of
parameters, you train a small set of additional "adapter" weights layered
on top of the frozen base model. QLoRA adds 4-bit quantization of the base
model during training, cutting memory needs further. The adapter alone is
tiny (megabytes, not gigabytes) until it's merged back into the base model
at the end.

### Full workflow

```
1. Curate a dataset
      │  instruction/output pairs, JSONL
      ▼
2. Pick training tooling + environment
      │  Unsloth (easiest) or Axolotl/HF peft (more control)
      │  needs a CUDA GPU — see hardware note below
      ▼
3. Run the LoRA/QLoRA training job
      │  produces a small adapter (safetensors), not a full model
      ▼
4. Merge the adapter into the base model weights
      │  produces one full-size fine-tuned model
      ▼
5. Convert to GGUF
      │  llama.cpp's convert script — GGUF is Ollama's serving format
      ▼
6. Write an Ollama Modelfile + `ollama create`
      │  registers the GGUF as a runnable Ollama model
      ▼
7. Point OLLAMA_MODEL at the new model name in .env
      │
      ▼
   pkmngpt's src/llm.py picks it up automatically — no code changes
```

#### Step 1 — Curate a dataset

This is the part most specific to pkmngpt. The dataset is instruction →
output pairs, e.g.:

```json
{"instruction": "What type is Charizard?", "output": "Charizard is a Fire/Flying type Pokemon."}
```

Sources worth drawing from:
- `eval/eval_set.json`, expanded well beyond its current 3-example
  placeholder — question/answer pairs you already trust.
- The live PokeAPI data `src/pokeapi.py` fetches — genus + flavor text per
  Pokemon can be turned into synthetic Q&A pairs ("What is X's Pokedex
  description?", "What type is X?", etc.) at some scale, since you have
  clean structured data underneath.
- Real transcripts from `scripts/ask.py` runs you've reviewed and judged
  correct — the best kind of training signal, since it reflects your
  actual pipeline's prompting style and citation format, not just raw
  facts.

A few hundred examples is a reasonable starting scale for LoRA on a
narrow domain like this; you don't need thousands.

#### Step 2 — Pick tooling + environment

- **Unsloth** — the easiest path for consumer hardware, optimized for
  speed/memory on single-GPU setups, good default choice to start with.
- **Axolotl** or raw HuggingFace `transformers` + `peft` — more
  configuration surface (YAML-driven pipelines), useful once you know
  what you're tuning and want more control.

**Hardware note:** this needs a CUDA GPU with roughly 8–16GB VRAM for
QLoRA on a 7–8B model (matching `llama3.1`'s scale). This tooling
ecosystem is CUDA-first — it does not run natively on Apple Silicon/Metal.
If your only hardware is a Mac, this step either needs a cloud GPU rental
(e.g. a single rented instance for the duration of a training run) or gets
deprioritized until you have GPU access. Worth confirming before investing
time in dataset curation.

#### Step 3 — Train the LoRA/QLoRA adapter

Supervised fine-tuning (SFT) on the instruction/output pairs from Step 1,
starting from the same base model Ollama already runs (e.g. `llama3.1`
instruct). Output is an adapter file (safetensors format) — small, and
separate from the base model weights until merged.

#### Step 4 — Merge the adapter into the base model

LoRA adapters are applied at inference time as a delta on top of the
frozen base weights; merging bakes that delta permanently into a
single full-size model. This step is required before GGUF conversion —
Ollama/llama.cpp need one merged model, not a base model plus a separate
adapter file.

#### Step 5 — Convert to GGUF

GGUF is llama.cpp's (and therefore Ollama's) model file format. The
merged, full-precision model gets converted and typically quantized down
(e.g. to 4-bit) for reasonable local inference speed/size — this is a
separate quantization decision from the QLoRA 4-bit *training*
quantization in Step 2, though both trade precision for size/speed.

#### Step 6 — Register it with Ollama

Write a `Modelfile` referencing the GGUF file, then:

```bash
ollama create pkmngpt-gen1 -f Modelfile
```

This makes `pkmngpt-gen1` runnable like any other Ollama model
(`ollama run pkmngpt-gen1`).

#### Step 7 — Wire it into pkmngpt

```bash
# .env
OLLAMA_MODEL=pkmngpt-gen1
```

`src/llm.py`'s `_generate_ollama()` already reads `config.OLLAMA_MODEL` —
no code changes needed on the pkmngpt side once the model is registered
with Ollama.

### Iteration loop

Unlike DSPy optimization (cheap to re-run, no GPU needed), a fine-tune is
a heavier, more occasional operation:

1. Ship the current fine-tune, use it for a while via `scripts/ask.py`.
2. Collect cases where it's wrong or where PokeAPI coverage grew (e.g.
   moving past gen 1).
3. Fold those into the dataset from Step 1.
4. Re-run Steps 3–6 to produce a new adapter/model version — don't retrain
   from scratch unnecessarily; treat each fine-tune as a versioned
   artifact (`pkmngpt-gen1-v2`, etc.) so you can roll back if a new
   version regresses.

---

## Open questions (decide before implementing)

- Where does the dataset-curation script live? Likely a new
  `scripts/`-level tool, separate from both the RAG pipeline and the
  fine-tuning job itself (which runs outside this repo, given the
  hardware/tooling requirements above).
- What counts as "good enough" to promote a new fine-tuned model version —
  presumably scored via the same DSPy `keyword`/`judge` metrics already
  used for the Claude path, run against the Ollama backend.
- Whether fine-tuning targets stay gen-1-only (matching current
  `src/pokeapi.py` scope) or get revisited once the project expands past
  gen 1.
