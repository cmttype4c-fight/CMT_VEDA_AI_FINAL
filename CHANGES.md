# What changed

## 1. UI — now matches the "CMT Veda AI" mockup
`static/index.html` was rebuilt as a dark two-pane layout:

- **Left sidebar** — Settings (Answer length: short/medium/long, Table
  formatting: auto/on/off, **User type: student/clinician/researcher — new**),
  a green "Knowledge base" stats box, and a collapsible "Indexed sources
  (masked)" list.
- **Right pane** — the "🕉 CMT Veda AI" header, the chat thread (question
  rows with a red avatar, answers with an orange avatar in a card,
  Markdown bold + table rendering, masked source chips), and the sticky
  input bar at the bottom.

All three settings are sent with every question and change how the
answer is written (see below) — nothing is just cosmetic.

New read-only endpoints power the sidebar:
- `GET /api/stats` → `{ chunk_count, document_count }`
- `GET /api/sources` → `{ sources: ["SRC-bf5632a8.pdf", ...] }`

Real filenames are **never** sent to the browser or shown to the model
inside citations — each document gets a stable masked id
(`SRC-<8 hex chars>.<ext>`, derived from a hash of its path) computed in
`main.py`. You don't need to re-run `ingest.py`; masking happens at
query time from the existing index.

## 2. Speed — ~10 min → ~30-50 sec per answer
The old code ran the 1.5B model through a plain `transformers` pipeline
on CPU, which is unoptimized and very slow for autoregressive
generation. It's been replaced with **`llama-cpp-python`** running a
**4-bit quantized GGUF** build of the same Qwen2.5-1.5B-Instruct model.
llama.cpp's CPU kernels + quantization are dramatically faster than
plain `transformers` on CPU-only servers, which is what gets a ~10
minute answer down to tens of seconds. Other contributing tweaks:

- `TOP_K` retrieval reduced 5 → 4 chunks, and each chunk truncated to
  1200 chars in the prompt (less input to process = faster).
- `max_new_tokens` now scales with the "answer length" setting instead
  of a fixed 250 (short answers finish much sooner).

### One-time setup on the server
```bash
pip install -r requirements.txt

pip install -U "huggingface_hub[cli]"
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --local-dir ./models
```
That's it — `main.py` looks for `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
by default. Override the path with the `RAG_GGUF_MODEL_PATH` env var if
you keep it elsewhere. `RAG_N_THREADS` defaults to all CPU cores; set it
explicitly if the box is shared with other services.

If your VPS has extra headroom and you want a quality bump at a modest
speed cost, `Qwen2.5-3B-Instruct-GGUF` (q4_k_m) is a drop-in swap — same
env var, just point it at that file instead.

## 3. Prompting per user type / length / table setting
`build_messages()` in `main.py` now composes the system prompt from
three presets (`USER_TYPE_PRESETS`, `LENGTH_PRESETS`,
`TABLE_INSTRUCTIONS`) so the same retrieved context is written up
differently depending on who's asking and what they asked for:

- **student** → plain language, terms defined on first use
- **clinician** → clinical/genetic terminology, diagnosis/management focus
- **researcher** → technical precision, mutation nomenclature, study detail

Edit those three dicts at the top of `main.py` if you want to tune the
wording further.
