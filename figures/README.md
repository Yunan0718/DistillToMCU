# Figures

`gen_all.py` regenerates every figure in the manuscript at publication quality:
vector PDF for LaTeX plus 300 DPI PNG for previews.

```bash
python figures/gen_all.py
```

The script reads the result files produced by the pipeline in `poc/`. Run the
experiments first, then regenerate the figures.

## Figure list

| File | Content | Manuscript |
|---|---|---|
| fig_architecture | System architecture: data source, PC-side distillation, MCU execution, cloud fallback | Figure 1 |
| fig_rule_lifecycle | Five-state rule lifecycle with transition thresholds | Figure 2 |
| fig_ar_learning | Autonomy growth per dataset, mean with bootstrap CI | Figure 3 |
| fig_baselines | Held-out autonomy and cloud-call reduction across baselines | Figure 4 |
| fig_risk_coverage | Coverage-risk operating points on the UCI replay | Figure 5 |
| fig_precision_recall | Teacher-replay precision and recall per dataset | Figure 6 |
| fig_cross_llm | Cross-teacher agreement and rule transfer | Figure 7 |
| fig_ablations | Distillation-source, generalization, lifecycle and selector ablations | Figure 8 |
| fig_latency | On-device match latency versus cloud round trip | Figure 9 |
| fig_nemenyi_cd | Friedman + Nemenyi critical-difference diagram | Supplementary |
| fig_oracle_vs_online | Online learning versus the full-information capacity ceiling | Supplementary |
| fig_rules_size | Rule-store growth over time | Supplementary |

`make_graphical_abstract.py` produces the graphical abstract (PDF, PNG, TIFF).