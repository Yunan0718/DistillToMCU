# Citation Audit Report — DistillToMCU

> academic-paper skill · Phase 5a (citation_compliance_agent) · Format: IEEE ·
> Draft: `paper/draft.md`

## Summary

| Metric | Value |
|--------|-------|
| Total in-text citations | 32 (numbered `[n]`) |
| Total reference list entries | 32 |
| Orphan in-text citations (no ref) | 0 |
| Orphan references (never cited) | 0 |
| Placeholder citations replaced | 8 (EdgeTalk, ESP-Claw, DCP, CIDER, HomeSGN, Hoffner, Darak, QuBan, MINTS) |
| Unverifiable references removed | 1 (Krentz, CC2538 TS) |
| Format errors auto-corrected | renumbered to order of appearance; sentence-case titles; 4 DOIs added |
| Items flagged for review | 4 (see below) |
| Missing DOIs | arXiv preprints (IDs given, no DOI exists); 2 GitHub repos (URLs given) |
| Self-citation ratio | 0% |
| Sources from 2022-2026 | 20/32 = 62.5% |

## Final Pass (2026-08-17)

Four issues were fixed across `draft.md`, `draft_cn.md`, and `main.tex`, and
the bibliography was recompiled:

| # | Location | Corrected | Basis |
|---|----------|-----------|-------|
| F1 | Sec. 4.3 | Added the Friedman/Nemenyi citation: J. Demšar, "Statistical comparisons of classifiers over multiple data sets," JMLR 7:1-30, 2006, now `[30]` | Verified via JMLR official bib (`Dem{\v{s}}ar`) |
| F2 | Ref [20] (MINTS) | Workshop name expanded: NeurIPS Workshop Math. Found. Oper. Integr. Mach. Learn. Uncertainty-Aware Decision-Making (MLxOR), 2025 | Verified via NeurIPS MLxOR workshop page |
| F3 | Ref [26] | "Grunwald" -> "Grünwald" (diacritic restored) | Verified via MIT Press record |
| F4 | Ref [30] -> [31] | Embedded Arena renumbered to `[31]` after inserting Demšar at `[30]` | IEEE first-appearance order |

## Corrections Made

| # | Location | Original | Corrected | Basis |
|---|----------|----------|-----------|-------|
| 1 | Whole draft | 30 refs, `[30]` out of order | Renumbered all in-text and list to first-appearance order (1-29) | IEEE rule: numbered in order of appearance |
| 2 | Ref [2] (EdgeTalk) | "[exact title pending verification]", MDPI | Full title, *Appl. Sci.* 16(12), doi 10.3390/app16125748 | Verified via publisher/DOI |
| 3 | Ref [4] (DCP) | "[exact title and authors pending]" | D. Yang, full title, arXiv:2605.26159 | Verified via arXiv |
| 4 | Ref [1] (HearthNet) | arXiv:2604.09618 | ACM CAIS 2026, doi 10.1145/3786335.3813188 | Verified via ACM DL / CAIS program |
| 5 | Ref [11] (ECS) | "X. Zhang, G. Wang, Y. Cui, et al." | 7 authors in full (Yanwei Cui, not Ya Cui) | Verified via dblp |
| 6 | Ref [6] (RIMRULE) | 5 names + et al. | 8 authors in full, pp. 34631-34646 | Verified via ACL Anthology |
| 7 | Ref [13] (CIDER) | "[exact full title pending]" | D. Jeong and H. Woo, full title, IEEE Access 13:197645-197662, 2025 | Verified via IEEE DOI |
| 8 | Ref [15] (HomeSGN) | "[exact title pending]" | Z. Yuan et al., full title, pp. 102-108, DOI | Verified via dblp / IEEE |
| 9 | Ref [16] | "Kaufman and Hoffner" | Y. Hoffner et al. (first author Hoffner), pp. 40-52 | Verified via dblp IoTBDS 2024 |
| 10 | Ref [18] (Darak) | "[exact title pending]" | S. V. S. Santosh and S. J. Darak, arXiv:2106.02855 | Verified via Semantic Scholar |
| 11 | Ref [19] (QuBan) | "[exact title pending]" | O. A. Hanna, L. Yang, C. Fragouli, PMLR v151, pp. 11215-11236 | Verified via dblp |
| 12 | Ref [20] (MINTS) | "in Proc. NeurIPS, 2025" | K. Wang, NeurIPS Workshop MLxOR, 2025 | Venue corrected (workshop, not main conference) |
| 13 | Deleted ref (Krentz) | "[bandits on CC2538 MCU], 2022" | Removed; Sec. 2.3 / 3.4 reworded to cite Darak only | Source not retrievable; IRON RULE: no unverifiable citations |
| 14 | Sec. 3.4 | "costs roughly 200 microseconds ... [17], [18]" | "impractical on microcontroller-class hardware ... [18]" | 200 us was an unmeasured code estimate, not a citable figure |
| 15 | Ref [17] | no DOI | doi: 10.1145/3088510 | Verified via ACM |
| 16 | Ref [23] | no DOI | doi: 10.1561/2200000101 | Verified via Now Publishers |
| 17 | Ref [28] | no DOI | doi: 10.1016/j.enbuild.2015.11.071 | Verified via Elsevier |
| 18 | Ref [29] | no DOI | doi: 10.1109/MC.2012.328 | Verified via IEEE |

## Items Flagged for Review

| # | Location | Issue | Suggested action |
|---|----------|-------|------------------|
| 1 | Ref [20] (MINTS) | Workshop name spelled "MLxOR" | Confirm the exact NeurIPS 2025 workshop title at LaTeX stage |
| 2 | Ref [16] (Hoffner et al.) | dblp lists the 5th author as "Fogel Harel" (likely reversed name) | Keep dblp order for now; verify on SCITEPRESS page if a reviewer asks |
| 3 | Refs [3], [12] (GitHub) | No retrieval date (IEEE convention: optional for software) | Add "[Online]. Available:" URLs already present; retrieval date only if journal requires |
| 4 | Classic refs [21]-[27] | No DOIs on Welford/Wilson/Kalai-Vempala/Greenwald/Gruenwald/Vovk | IEEE commonly omits DOIs for pre-2005 classics; acceptable, flagged for editorial check |

## Notes

- "AFD-KD venue = ACM" appears in the Phase 0 configuration record but AFD-KD is
  not cited in the current draft, so no correction was needed.
- STRANDS data lineage: the draft cites CASAS (Cook et al.) as the source
  dataset; the numeric sensor fields are synthetic completions, stated in
  Sections 4.1, 5.2, and 7. The Zenodo record 17180309 (CASAS Smart Home
  dataset, Cook, 2025, doi 10.5281/zenodo.17180309) was added as `[32]` and
  cited in the Data Availability statement.
- The WireClaw entry `[12]` now lists the author as "Open-source project (no
  listed authors)" instead of the GitHub handle.
- All 32 citations are cross-checked: every in-text `[n]` has a reference and
  every reference is cited at least once; numbering is strictly by first
  appearance.

## v10.7 Update (2026-08-19)

The three new real datasets (SML2010, Steel, Air Quality) added three
citations and two fixes were applied across `draft.md`, `main.tex`, and
`draft_cn.md`:

| # | Location | Change | Basis |
|---|----------|--------|-------|
| A | Ref [2] | Title prefix "EdgeTalk-MCU:" restored | MDPI article page (10.3390/app16125748) |
| B | Ref [16] | Author list verified as 5 authors (Y. Hoffner, E. Kaufman, A. Amir, E. Yovel, F. Harel); earlier 3-author note was an error | dblp IoTBDS 2024: Yigal Hoffner, Eran Kaufman, Avidan Amir, Elad Yovel, Fogel Harel, pp. 40-52 |
| C | Ref [21] | doi 10.1016/j.jcss.2004.10.016 added | Crossref / ScienceDirect |
| D | Ref [25] | doi 10.1080/00401706.1962.10490022 added | Crossref / T&F |
| E | Ref [27] | doi 10.1080/01621459.1927.10502953 added | Crossref / T&F |
| F | Ref [30] | URL https://jmlr.org/papers/v7/demsar06a.html added | JMLR official page |
| G | Refs [33]-[35] | New dataset citations: Zamora-Martinez 2014 (SML2010), Sathishkumar 2021 (Steel), De Vito 2008 (Air Quality) | UCI repository + Crossref |

Total reference list entries: 35. All 35 were re-verified this round against
Crossref/publisher/arXiv/dblp records (titles, author order, venue, volume,
pages, DOI); no orphan citations and no uncited entries. Items flagged above
(1)-(4) remain editorial-only checks.
