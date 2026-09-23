# AI Control & Framework Benchmarking Tool

Maps an organisation's own AI/technology control library against reference
frameworks — NIST AI RMF, EU AI Act high-risk system obligations, and
ISO/IEC 42001-style AI management system topic areas — and flags framework
requirements that don't have a confident match in the control library, so a
human can review them.

This is a self-contained, public version of the kind of control-benchmarking
approach used in AI risk governance work: instead of manually cross-walking
a control library against a new regulatory framework line-by-line, the tool
scores every framework requirement against your closest matching internal
control and surfaces where coverage looks weak, absent, or uncertain.

## Why this exists

Manual control-mapping exercises (e.g. "does our control library cover the
EU AI Act's Article 14 human oversight requirement?") are slow and easy to
get subtly wrong — synonyms and paraphrased requirements get missed by
keyword search, and full coverage reviews don't scale as frameworks
multiply (NIST, EU AI Act, ISO 42001, and more all landing on
organisations at once).

## How it works

1. Your control library is a simple CSV (`control_id`, `description`).
2. Each reference framework is a JSON list of control-like requirements.
3. A similarity engine scores every framework requirement against every
   company control and keeps the best match.
4. Anything below a calibrated similarity threshold is flagged
   `NEEDS_REVIEW`; everything else is `COVERED`.

**This tool deliberately never auto-confirms a gap or a match.** Extensive
calibration (see [`docs/findings.md`](docs/findings.md) for the full
investigation) showed that even a strong similarity engine can produce a
*confident, wrong* match — some company controls act as generic "attractors"
that score plausibly against almost any requirement without being truly
relevant. `COVERED` here means "the tool's best guess, worth spot-checking,"
not "confirmed adequate" — every result includes the actual matched control
text so a human can verify it, and results are sorted lowest-score-first so
the most doubtful ones surface first.

The similarity backend is pluggable (`similarity_engine.py`), selectable via
`--engine`:

- **`tfidf`** (default) — pure lexical/keyword similarity via TF-IDF +
  cosine similarity. Runs fully offline, no API key or model download
  required. Good for proving the pipeline end-to-end; weak at catching
  paraphrased matches (e.g. "user can override outputs" vs. "human
  oversight" score poorly despite meaning almost the same thing).
- **`embedding`** — a local `sentence-transformers` model
  (`all-MiniLM-L6-v2` by default). Catches paraphrased matches TF-IDF
  misses, but calibration found its raw cosine similarity scores aren't
  reliably comparable to a single threshold — see the findings doc for why.
- **`cross-encoder`** — a `sentence-transformers` `CrossEncoder`
  (`cross-encoder/stsb-roberta-base` by default) that scores each
  requirement/control pair jointly rather than comparing independent
  embeddings. The most accurate of the three in testing, but still not
  reliable enough to trust unsupervised — see findings.

Each engine has its own calibrated default `--threshold`
(`DEFAULT_THRESHOLDS` in `benchmark.py`), since raw scores are not
comparable across engines.

## Quickstart

```bash
pip install -r requirements.txt

python benchmark.py --controls data/sample_company_controls.csv --framework nist
python benchmark.py --controls data/sample_company_controls.csv --framework all --engine cross-encoder
```

Output: a console summary per framework, plus a combined CSV report at
`output/benchmark_report.csv` with every framework control, its best
match, match score, and `COVERED` / `NEEDS_REVIEW` status.

## Important note on the reference framework data

The JSON files under `frameworks/` are **illustrative, paraphrased summaries**
written for this demo — not verbatim regulatory or standard text. NIST AI
RMF content is US-government public domain, so it can be adapted freely;
EU AI Act obligations are paraphrased from public legislation; the ISO
42001 file lists generic AI management system *topic areas* only, since
ISO standard text itself is copyrighted and not reproduced here. **Do not
use these files as an actual compliance reference** — pull the real
framework text from the official sources before using this for anything
beyond a portfolio demo.

## Roadmap / ways to extend this

- [ ] Try LLM-as-judge scoring for small control libraries (cheap and likely
      more accurate at this scale — no training data needed, unlike a
      fine-tuned embedding model; see findings doc for the scale tradeoffs).
- [ ] Detect and flag "hub" controls — ones that win an unusually high
      share of best-match slots across unrelated requirements — as a
      targeted mitigation for the attractor problem found during testing.
- [ ] Add a simple Streamlit front-end for uploading a control CSV and
      viewing results interactively instead of via CLI.
- [ ] Surface the second-best match alongside the best whenever the margin
      between them is small, instead of silently picking one.
- [ ] Add a second framework input mode: instead of `all`, benchmark
      the *company's* controls for orphans (controls that don't map to
      any framework requirement — the inverse gap analysis).
- [ ] An agentic version that scrapes for regulatory/standard updates and
      flags their impact on an existing control library is a natural next
      project — but only as a recommend-and-review tool, never one that
      edits a live control library unsupervised (see findings doc).

## Why I built this

Built as a public demonstration of an approach used in AI risk governance
work — mapping control libraries against evolving AI regulatory and
standards frameworks (NIST AI RMF, EU AI Act, ISO/IEC 42001) at speed as
new requirements land on organisations.
