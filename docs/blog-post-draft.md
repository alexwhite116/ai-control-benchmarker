# Why I gave up trying to fully automate AI control-mapping

Mapping a company's AI controls against a regulatory framework — does our
control library actually cover the EU AI Act's human oversight requirement?
— is normally a manual, line-by-line exercise. It's slow, and with NIST AI
RMF, ISO 42001, and whatever comes next all landing at once, it doesn't
scale.

I built a small tool to automate the first pass: score a company's
controls against a framework's requirements and flag anything without a
good match. Getting it to run was easy. Getting it to a point I'd trust
its output took a lot longer, and most of that time went into finding out
how many ways it could be confidently wrong.

## Keyword matching doesn't get paraphrasing

Baseline is TF-IDF plus cosine similarity — literal keyword overlap. It
fails on paraphrasing: "users can escalate or override AI outputs" and
"the system supports human oversight, including the ability to intervene"
mean roughly the same thing and score near zero, because they don't share
words.

Embeddings are the standard fix — encode meaning into a vector instead of
counting words. I swapped in a local sentence-transformers model expecting
it to mostly solve this.

## Embeddings pick up jargon, not just meaning

I hand-picked six pairs to check the model before trusting it on real
data — three obvious matches, three obvious non-matches. Two of the
non-matches scored higher than one of the real matches.

The pattern: every pair except one used the word "AI" on both sides. The
one pair that separated cleanly was the one where neither side did. The
model wasn't only tracking meaning, it was tracking "these are both about
the same general topic" — not the same thing as "these say the same
thing." A control about incident-response playbooks and a requirement
about diversity and inclusion in AI risk management both mention "AI risk
management" enough to look moderately related, without having anything
actually in common.

Sentence embeddings are known to cluster in a narrow region of the vector
space rather than spreading toward zero for unrelated text. In a
jargon-dense domain where nearly every sentence says "AI," "risk," or
"system," that clustering is a real problem, not a footnote.

A bigger, generally stronger model didn't fix it — everything scored
higher, but the overlap between match and non-match stayed the same width.
So it wasn't a "too small" problem.

## A free heuristic that worked on six pairs and broke on fourteen

Instead of an absolute threshold, I tried scoring each match by how much
it stood out from the other nine candidates for that requirement — a
z-score against the row's own mean and standard deviation, using data
already computed, no extra labels needed.

On the original six pairs it worked cleanly: every match outranked every
non-match. I expanded to fourteen pairs to stress-test it properly, and
the separation broke. One broadly-worded internal audit control stood out
from its row even against a requirement it had nothing to do with, purely
because it read as plausible next to whatever else was on the page.
Standing out from nine bad candidates isn't the same as being right.

## Cross-encoders help, then expose a bigger problem at scale

A cross-encoder reads both sentences together and scores the pair jointly,
instead of comparing two independently-built vectors — harder to fool with
shared vocabulary in theory, since it compares content rather than topic.

On the fourteen pairs it did better: 12/14 separated correctly, versus
roughly 11/14 for the earlier methods. I adopted it and ran the real
version — all 42 requirements across three frameworks against all 10
sample controls, not just my hand-picked pairs.

One company control — a logging and traceability control — got picked as
the best match for 20 of the 42 requirements. Diversity and inclusion
processes. Third-party risk mapping. Stakeholder feedback. None of it
related to logging. My calibration pairs never caught this because they
only ever tested one control against one requirement at a time. The real
algorithm picks a winner out of ten competing candidates every time, and a
control with generic, plausible-sounding phrasing can look like a
reasonable answer to almost anything — the same "hubness" effect seen in
nearest-neighbor search, where certain points end up as disproportionately
frequent matches for unrelated queries.

I tried scoring by margin next — how much the winner beat the runner-up
by, on the theory that a real match should win decisively. Worked on
average. Also had a clean counterexample: the same logging control beat
its runner-up by a healthy margin on a requirement it had nothing to do
with, because the runner-up wasn't a good match either.

## Where it landed

Four scoring methods, four confirmed ways to be confidently wrong. That's
not a reason to try a fifth. The tool now never asserts a confirmed gap or
match — every result is a best guess with the matched text shown alongside
it, sorted worst-first, for a human to check. Given all of the above,
that's not a hedge, it's an accurate description of what the model can
actually do.

(There's an obvious joke here about a tool meant to check human oversight
requirements needing human oversight itself. I'll leave it at that.)

## What's next

Fine-tuning a domain-specific embedding model would need real training
volume I don't have, and generating that via an LLM raises the obvious
question of why not just ask the LLM directly. At this scale — a company's
control library, not millions of documents — that's probably both simpler
and more accurate, and it's next on the list.

I didn't build the version that scrapes for regulatory updates and edits a
control library automatically, either. Letting something act unsupervised
on a signal I've now shown four different ways to be wrong seemed like
exactly the wrong takeaway from this project.

Code and the full calibration data are on GitHub:
[ai-control-benchmarker](https://github.com/alexwhite116/ai-control-benchmarker).
