# Findings: what actually happened trying to make this reliable

*Draft notes from building the embeddings/cross-encoder backend for the AI
Control & Framework Benchmarking Tool. Written as source material for a
write-up — reads as engineering notes, not finished prose. Numbers are from
real runs against this repo's sample data (`data/sample_company_controls.csv`
vs `frameworks/*.json`), not simulated.*

## Baseline: TF-IDF keyword overlap
The TF-IDF baseline proves the pipeline works but
misses paraphrases by design. "User can escalate or override outputs"
and "human oversight... ability to intervene" mean almost the same thing
but score close to zero based on shared vocabulary alone. The obvious fix to be attempted next is to
swap in real embeddings and capture semantic similarity.

## Attempt 1: bi-encoder embeddings (`all-MiniLM-L6-v2`)

6 calibration pairs were hand-picked: 3 that should clearly match, 3 that
clearly shouldn't. We then checked where the model actually placed them:

| Pair | Expected | Raw cosine score |
|---|---|---|
| CTL-007 ↔ EUAI-6 (override outputs / human oversight) | match | 0.453 |
| CTL-010 ↔ ISO-6 (training / competence) | match | 0.577 |
| CTL-006 ↔ EUAI-2 (data quality / data governance) | match | 0.386 |
| CTL-002 ↔ GOVERN-3 (performance metrics / DEI) | non-match | 0.092 |
| CTL-004 ↔ MEASURE-4 (vendor questionnaire / fairness) | non-match | **0.417** |
| CTL-009 ↔ GOVERN-3 (incident playbook / DEI) | non-match | **0.384** |

Two non-matches scored *higher* than a genuine match. Every pair except
one shares the literal word "AI" on both sides; the one clean outlier
(0.092) is the only pair where one side never says "AI" at all. The initial hypothesis is that the model isn't heavily favouring similarity based on shared vocabulary rather than purely semantic meaning. 

Sentence embeddings are known to cluster in a narrow region of the vector space
(anisotropy) rather than spreading toward 0 for "unrelated," so
domain-adjacent-but-different sentences score much higher than intuition
suggests.

## Attempt 2: a bigger model doesn't fix it

Same 6 pairs against `all-mpnet-base-v2` (larger, generally stronger):

| | min match score | max non-match score |
|---|---|---|
| MiniLM (small) | 0.386 | 0.417 |
| mpnet (large) | 0.496 | 0.498 |

Everything shifted up, but the *overlap* between match and non-match remained. Confirms this isn't "small model = weak model" — it's structural
to comparing short, domain-narrow sentences via raw cosine similarity,
regardless of size.

## Attempt 3: self-calibrated z-score

Instead of using an absolute threshold, score each match by how much it stands
out from the *other 9 company controls'* scores for that same requirement
(z-score against the row's own mean/std) with no external calibration data
needed, since the full candidate set is already computed. Initial hypothesis was that this would not work very well as it misses out on what happens if *none* of the controls cleanly map to one another (it only cares about what the *best* mapping relative to the others is).

On the original 6 pairs this appeared to work cleanly: all 3 matches ranked above all
3 non-matches. However, when expanded to 14 pairs (covering all 10 company controls, all
3 frameworks) to stress-test it, the separation broke:

- min match z-score: 0.334
- max non-match z-score: **0.947** (`CTL-008 ↔ EUAI-5`)

`CTL-008` ("internal audit... reviews AI systems for conformance") is a
broad, generally-applicable-sounding control. It stood out from its row
even against a requirement it doesn't actually address. This is a result of the initial hypothesised weakness of this approach - generally-applicable-sounding controls will seem much better than other, more specific controls, even if semantically they are still not fit for purpose.
## Attempt 4: cross-encoder reranking

Switched to `cross-encoder/stsb-roberta-base`. This scores both texts jointly
instead of comparing independently-computed embeddings, which should be
less fooled by shared vocabulary since it reads both sentences together.

Same 14 pairs:

| | min match | max non-match | separated correctly |
|---|---|---|---|
| Raw MiniLM cosine | 0.386 | 0.417 | 11/14 |
| Self-calibrated z-score | 0.334 | 0.947 | 11/14 |
| Cross-encoder | 0.283 | 0.351 | **12/14** |

This initially appeared to be the best of the three, and the one remaining miss was a different failure mode: under-scoring a genuine paraphrase (staff training vs. personnel
competence) rather than being fooled by shared vocabulary. This was encouraging as for a compliance-critical application such as this, we are significantly more willing to accept false negatives than false positives. Adopted this as the
production scoring engine for a full test.
## The real test: full-scale run surfaces a worse problem

Running all 42 framework requirements against all 10 company controls (420
pairs compared to the previous 14 sample pairs) resulted in markedly worse performance once again:

| Company control | # requirements it "won" | Actually relevant (manual review) |
|---|---|---|
| CTL-003 (logs/traceability) | 20 | 1 (EUAI-4, automatic event recording) |
| CTL-008 (internal audit) | 8 | 1 strong (ISO-11) + 1 weak (ISO-12) |
| CTL-009 (incident response) | 4 | 1 (MANAGE-4) |

CTL-003 was picked as the "best match" for 20 of 42 requirements, including
DEI processes, third-party risk mapping, and stakeholder feedback
mechanisms, none of which it addresses. Its phrasing ("system events,"
"post-incident review," "AI-enabled tools") is broad enough to sound
plausibly relevant to almost anything AI-governance-shaped.

**This leads to the conclusion that there is a gap in calibration methodology, rather than just a model weakness.**
Isolated pairwise testing (does pair X look related?) doesn't accurately test what
the algorithm actually does: pick one winner out of 10 competing candidates
per requirement, assuming at least one control is 'enough' of a match. Multiple controls can be considered plausible but our calibration did not account for how the model comes to a judgment as to which is the most plausible. This leads to the phenomenon
of *hubness* as in nearest-neighbor search, where certain points become
disproportionately frequent "nearest neighbors" to many unrelated queries
because of some property of the point itself, as opposed to genuine relevance.

## Attempt 5: does winning margin help?

We tried to test whether a significant gap between the best and second-best score
(`best_score - second_best_score`) distinguishes real matches from hub
false-positives. The idea was that a genuine match should win decisively, whereas 'hub' controls
might score highly but would not necessarily significantly outperform their peers.
- known-good matches: mean margin **0.175**
- known-spurious matches: mean margin **0.055**

On average, this seemed to produce a decent result but significant outliers mean it is unrealiable: `GOVERN-5 ↔ CTL-003` (spurious) scored 0.616
with a margin of 0.182, beating 4 of the 7 genuine matches on *both* axes.
CTL-003 beat its own runner-up decisively while still being wrong, because
the runner-up (`CTL-007`) wasn't a good match either. The margin measures whether the winner beat the field but not whether the winner was correct. Combined with the generally poor performance of the similarity scores intended to determine precisely that correctness, it was decided that this was another dead end.
## Where this landed

Four different signals (raw cosine, self-calibrated z-score, cross-encoder
absolute score, and winning margin) each showed a real, reproducible
failure on this dataset. This suggests that no single scalar similarity score reliably classifies match
quality here, at least not with general-purpose (not domain fine-tuned)
models.

**Design conclusion:** the tool doesn't auto-confirm coverage. `COVERED`
reflects that this control is more likely to be covered but that this requires human review. Every result
carries the actual matched text so a reviewer can check it, and results
are sorted lowest-score-first so the most doubtful cases are surfaced at a glance.
## Considered and deliberately not built (yet)

**Fine-tuning a domain-specific embedding model.** Would require hundreds to thousands of labeled pairs, not the ~20 hand-built
here; with this little data a fine-tuned model would memorise these exact
sentences rather than generalize. Generating that volume via LLM-labeled
synthetic pairs raises an obvious question: why not use an LLM directly?

**Similar solution using LLM calls**: Instead of running through an embedding model and associated similairty engine, prompt an LLM to provide a judgement on which controls map and the level of coverage. This is a natural extension of this solution and would not require significant architectural rework to get working, but has been scoped out for now due to complexity of obtaining an API key or running a local model. This is firmly on the list for future consideration in the backlog.

**A fully autonomous agentic version**: Scraping for regulatory
updates and making edits to the control library directly, not just recommendations, was deliberately scoped out. Everything above demonstrates that even a
reasonably sophisticated similarity signal can be confidently wrong. Letting
an agent autonomously make changes to a real compliance artifact on that same class
of signal, with no human review, risks significant mistakes infiltrating governance practices and exposing an organisation to regulatory breaches. There's also an additional as-yet-unresolved provenance problem: this project's reference framework files are paraphrased summaries, not verbatim
regulatory text, and a scraper would need to solve "is this actually the
authoritative source" before its output is trustworthy input to anything.
The recommend-and-review version of that idea is a reasonable future
project which is also on the list for backlog consideration. The autonomous control library maintenance and change (without human review) tool is not.
