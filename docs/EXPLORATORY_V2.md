# Exploratory standard/v2 adapter

`quant_lab.research_v2.write_exploratory_run_v2` publishes daily NAV and optional
long positions with the research profile. The adapter always marks results as
non-investable and non-rankable; it does not certify accounting or a historical
universe. Date-only records use the end of their UTC day. Supply aware timestamps
when an exact market close is known. Quantities and amounts use four decimal
places, and dataset snapshots must identify input content, not a directory name.

`clean_git_commit` returns a revision only when Git successfully verifies a clean
checkout. Callers can retain their existing reports when no clean revision is
available. Consumers must pin a quant-lab revision containing this module in both
their package dependencies and their lock files.

The scanner excludes research profiles, legacy contracts, explicit watchlists and
non-investable runs from certified rankings. Research result comparisons remain
separate: an explicit `rankable=false` or `standard/v1` label prevents comparison,
while older research result payloads retain their existing compatibility checks.
