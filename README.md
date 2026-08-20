# AI Control & Framework Benchmarking Tool

Maps an organisation's own AI/technology control library against reference
frameworks — NIST AI RMF, EU AI Act high-risk system obligations, and
ISO/IEC 42001-style AI management system topic areas — and automatically
flags gaps where no adequately similar internal control exists.

This is a self-contained, public version of the kind of control-benchmarking
approach used in AI risk governance work: instead of manually cross-walking
a control library against a new regulatory framework line-by-line, the tool
scores every framework requirement against your closest matching internal
control and highlights where coverage is weak or absent.

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
4. Anything below a configurable similarity threshold is flagged as a gap.

The similarity backend is intentionally pluggable (see
`similarity_engine.py`):

- **`TfidfSimilarityEngine`** (default, included) — pure lexical/keyword
  similarity via TF-IDF + cosine similarity. Runs fully offline, no API
  key required. Good for proving the pipeline, weak at catching
  paraphrased matches (e.g. "user can override outputs" vs. "human
  oversight" score poorly despite meaning almost the same thing).
- **`EmbeddingSimilarityEngine`** (stub, ready to fill in) — swap in
  Azure OpenAI, OpenAI, or a local `sentence-transformers` model and the
  rest of the codebase is unchanged. This is the natural next step and
  would meaningfully improve match quality on real-world control language.

## Quickstart

```bash
pip install -r requirements.txt

python benchmark.py --controls data/sample_company_controls.csv --framework nist
python benchmark.py --controls data/sample_company_controls.csv --framework all --threshold 0.15
```

Output: a console summary per framework, plus a combined CSV report at
`output/benchmark_report.csv` with every framework control, its best
match, match score, and gap/covered status.

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

- [ ] Wire up `EmbeddingSimilarityEngine` to Azure OpenAI or
      `sentence-transformers` and compare match quality against the
      TF-IDF baseline (good writeup material on its own).
- [ ] Add a simple Streamlit front-end for uploading a control CSV and
      viewing results interactively instead of via CLI.
- [ ] Add confidence bands / multiple candidate matches per framework
      control, not just the single best match.
- [ ] Add a second framework input mode: instead of `all`, benchmark
      the *company's* controls for orphans (controls that don't map to
      any framework requirement — the inverse gap analysis).

## Why I built this

Built as a public demonstration of an approach used in AI risk governance
work — mapping control libraries against evolving AI regulatory and
standards frameworks (NIST AI RMF, EU AI Act, ISO/IEC 42001) at speed as
new requirements land on organisations.
