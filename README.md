# AI Control & Framework Benchmarking Tool

Maps an organisation's own AI/technology control library against reference
frameworks — NIST AI RMF, EU AI Act high-risk system obligations, and
ISO/IEC 42001-style AI management system topic areas — and flags framework
requirements that don't have a confident match in the control library, so
a human can review them.

This is a self-contained, public version of the kind of control-benchmarking
approach used in AI risk governance work. Rather than manually cross-walking
a control library against a new regulatory framework line-by-line, the tool
scores every framework requirement against the closest matching internal
control and highlights where coverage looks weak, absent, or uncertain.

## Why this exists

Manual control-mapping exercises (e.g. "does our control library cover the
EU AI Act's Article 14 human oversight requirement?") are slow and easy to
get subtly wrong. Synonyms and paraphrased requirements get missed by
keyword search, and full coverage reviews don't scale as frameworks
multiply (NIST, EU AI Act, ISO 42001, and more all landing on
organisations at once).

## How it works

1. The control library is a simple CSV (`control_id`, `description`).
2. Each reference framework is a JSON list of control-like requirements.
3. A similarity engine scores every framework requirement against every
   company control and keeps the best match.
4. Anything below a calibrated similarity threshold is flagged
   `NEEDS_REVIEW`; everything else is `LIKELY_MATCH`.

**The tool deliberately never auto-confirms a gap or a match.** Extensive
calibration (see [`docs/findings.md`](docs/findings.md) for the full
write-up) found that even a strong similarity engine can produce a
confident, wrong match — some company controls act as generic "attractors"
that score plausibly against almost any requirement without actually being
relevant to it. `LIKELY_MATCH` means "the tool's best guess, worth
spot-checking", not "confirmed adequate" — the status was named this way
deliberately, after "COVERED" turned out to overclaim. Every result
includes the matched control text so a human can verify it, and results
are sorted lowest-score-first so the most doubtful ones are seen first.

The similarity backend is pluggable (`similarity_engine.py`), selectable
via `--engine`:

- **`tfidf`** (default, because it runs fully offline with no model
  download or API key — not because it's the most accurate) — plain
  lexical/keyword similarity via TF-IDF and cosine similarity. Good enough
  to prove the pipeline end-to-end, but weak at catching paraphrased
  matches (e.g. "user can override outputs" vs. "human oversight" score
  poorly despite meaning almost the same thing).
- **`embedding`** — a local `sentence-transformers` model
  (`all-MiniLM-L6-v2` by default). Catches paraphrased matches TF-IDF
  misses, but calibration found its raw cosine similarity scores aren't
  reliably comparable against a single threshold — see the findings doc
  for why.
- **`cross-encoder`** — a `sentence-transformers` `CrossEncoder`
  (`cross-encoder/stsb-roberta-base` by default), which scores each
  requirement/control pair jointly rather than comparing independent
  embeddings. The most accurate of the three during testing, but still
  not reliable enough to trust unsupervised (see findings doc).

Each engine has its own calibrated default `--threshold`
(`DEFAULT_THRESHOLDS` in `benchmark.py`), since raw scores aren't
comparable across engines.

## Quickstart

```bash
pip install -r requirements.txt

python benchmark.py --controls data/sample_company_controls.csv --framework nist
python benchmark.py --controls data/sample_company_controls.csv --framework all --engine cross-encoder
```

Output: a console summary per framework, plus a combined CSV report at
`output/benchmark_report.csv` with every framework control, its best
match, match score, and `LIKELY_MATCH` / `NEEDS_REVIEW` status.

**Note on accuracy:** no engine here has measured precision/recall against
ground truth yet — every number in `docs/findings.md` so far is from small
hand-picked calibration pairs, not the full 42-requirement set. Treat
`--engine cross-encoder` as "the most promising in early testing," not
"the validated choice", until that work is done.

## Important note on the reference framework data

The JSON files under `frameworks/` are **illustrative, paraphrased
summaries** written for this demo, not verbatim regulatory or standard
text. NIST AI RMF content is US-government public domain, so it can be
adapted freely; EU AI Act obligations are paraphrased from public
legislation; the ISO 42001 file lists generic AI management system *topic
areas* only, since ISO standard text itself is copyrighted and not
reproduced here. **Do not use these files as an actual compliance
reference** — pull the real framework text from the official sources
before using this for anything beyond a portfolio demo.

## Roadmap

Deliberately short. Earlier drafts of this list included a Streamlit
front-end, a second-best-match display, and an orphan-control analysis
mode — cut because none of them would teach anything new or change what a
reviewer concludes about the actual hard problem here, which is match
quality, not presentation.

- [ ] Hand-label all 42 framework requirements with their correct company
      control (or "no adequate control exists") and measure actual
      precision/recall of `LIKELY_MATCH` per engine. Nothing above this
      line in the project is validated without it.
- [ ] Fix the self-calibrated z-score properly: it only normalises each
      requirement's row (control vs. the other 9 candidates), which can't
      penalise a hub control, since a hub scores highly in every row it's
      in. The standard correction for this — from the unsupervised
      word-translation literature, which hits the same hubness problem —
      is CSLS (Cross-domain Similarity Local Scaling): penalise a control
      by its own average similarity across *all* requirements, not just
      within one row.
- [ ] Build the LLM-as-judge engine and compare it against the above with
      real ground truth, not just calibration pairs.
- [ ] Test a reranker trained for query-to-passage relevance (an MS MARCO
      cross-encoder, or an NLI model) instead of `stsb-roberta-base`, which
      is trained for symmetric "do these mean the same thing" similarity —
      a different question from "does this control satisfy this
      requirement".
- [ ] An agentic version that scrapes for regulatory/standard updates and
      flags their impact on an existing control library is a reasonable
      future project — but only as a recommend-and-review tool, never one
      that edits a live control library unsupervised (see findings doc).

## Why I built this

Built as a public demonstration of an approach used in AI risk governance
work: mapping control libraries against evolving AI regulatory and
standards frameworks (NIST AI RMF, EU AI Act, ISO/IEC 42001) at speed as
new requirements land on organisations.
