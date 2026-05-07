# Appendix Hypothesis-Test Targets and Metric-Consistency Notes

Sources:

- `benchmarks/tso/results/summary/latest.json`
- `benchmarks/dso/results/summary/latest.json`
- `benchmarks/ders/results/summary/latest.json`
- `benchmarks/gencos/results/summary/latest.json`
- `benchmarks/dc_microgrid/results/summary/latest.json`

This appendix table summarizes existing `latest.json` outputs only. It does not alter result values.

## Hypothesis-Test Target Table

| task | test name | metric | primary split | comparison/null target | n_pairs or paired seeds | p-value/effect direction | whether used for claim eligibility |
|---|---|---|---|---|---|---|---|
| tso | paired_signflip_permutation | total_operating_cost | iid | ppo_lagrangian vs all_on; H0: paired left-right cost difference = 0 | n_pairs=250; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-900508.3235; lower_is_better => ppo_lagrangian better | no: cost-only evidence; hard safety gates are evaluated separately and no IID row passes both |
| tso | paired_signflip_permutation | total_operating_cost | iid | ppo_lagrangian vs ppo; H0: paired left-right cost difference = 0 | n_pairs=250; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=1854773.556; lower_is_better => ppo_lagrangian worse | no: cost-only evidence; hard safety gates are evaluated separately and no IID row passes both |
| tso | paired_signflip_permutation | total_operating_cost | iid | ppo vs all_on; H0: paired left-right cost difference = 0 | n_pairs=250; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-2755281.8795; lower_is_better => ppo better | no: cost-only evidence; hard safety gates are evaluated separately and no IID row passes both |
| dso | paired_signflip_permutation | total_loss_mwh | iid | ppo vs no_control; H0: paired seed-level loss difference = 0 | paired_seeds=[0,1,2,3,4]; n_pairs=5 | two-sided p=0.0625; mean_diff=-0.9554199707508086; lower_is_better => ppo better | yes: primary loss target, with n=5 seed-pair p-value floor caveat |
| dso | paired_signflip_permutation | total_loss_mwh | iid | sac vs no_control; H0: paired seed-level loss difference = 0 | paired_seeds=[0,1,2,3,4]; n_pairs=5 | two-sided p=0.0625; mean_diff=-0.5730319805145264; lower_is_better => sac better | yes: primary loss target, with n=5 seed-pair p-value floor caveat |
| dso | paired_signflip_permutation | total_loss_mwh | iid | saute_ppo vs no_control; H0: paired seed-level loss difference = 0 | paired_seeds=[0,1,2,3,4]; n_pairs=5 | two-sided p=0.0625; mean_diff=-0.9538597621917724; lower_is_better => saute_ppo better | yes: primary loss target, with n=5 seed-pair p-value floor caveat |
| dso | paired_signflip_permutation | total_loss_mwh | iid | ppo_lagrangian vs no_control; H0: paired seed-level loss difference = 0 | paired_seeds=[0,1,2,3,4]; n_pairs=5 | two-sided p=0.0625; mean_diff=-0.4800828862190246; lower_is_better => ppo_lagrangian better | yes: primary loss target, with n=5 seed-pair p-value floor caveat |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo vs no_control; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-0.007759741594394048; lower_is_better => ippo better | yes: primary IID loss target |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo vs volt_droop; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-0.005726907799641291; lower_is_better => ippo better | yes: primary IID loss target |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo_safe vs no_control; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-0.00777787779768308; lower_is_better => ippo_safe better | yes: primary IID loss target |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo_safe vs volt_droop; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-0.005745044002930323; lower_is_better => ippo_safe better | yes: primary IID loss target |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo_lagrangian vs no_control; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=0.8699065046747663; mean_left_minus_right=-0.00007029970486958822; lower_is_better => ippo_lagrangian slightly better, not significant | yes: primary IID loss target, but neutral/non-headline evidence |
| ders | paired_signflip_permutation | mean_p_loss_mw | iid | ippo_lagrangian vs volt_droop; H0: paired left-right mean_p_loss_mw difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=0.0019625340898831683; lower_is_better => ippo_lagrangian worse | yes: primary IID loss target, as negative evidence |
| gencos | paired_signflip_permutation | total_profit | iid | ippo vs truthful; H0: paired left-right total_profit difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=388154.56447419483; higher_is_better => ippo better | yes: primary profit claim, positive against truthful |
| gencos | paired_signflip_permutation | total_profit | iid | ippo vs uniform_mid; H0: paired left-right total_profit difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=92116.07971182505; higher_is_better => ippo better | yes: primary profit claim, positive against uniform_mid |
| gencos | paired_signflip_permutation | total_profit | iid | ippo vs max_markup; H0: paired left-right total_profit difference = 0 | n_pairs=150; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=-202775.38020600638; higher_is_better => ippo worse | yes: primary profit target, as negative evidence against the strong max_markup heuristic |
| dc_microgrid | paired_signflip_permutation | episode_reward | iid | sac vs ppo; H0: paired left-right episode_reward difference = 0 | n_pairs=50; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=41.617351211607456; higher_is_better => sac better | yes: primary reward claim only; feasibility audit remains separate |
| dc_microgrid | paired_signflip_permutation | episode_reward | iid | ppo vs rule_based; H0: paired left-right episode_reward difference = 0 | n_pairs=50; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=33851.14722388953; higher_is_better => ppo better | yes: primary reward claim only; feasibility audit remains separate |
| dc_microgrid | paired_signflip_permutation | episode_reward | iid | sac vs rule_based; H0: paired left-right episode_reward difference = 0 | n_pairs=50; common_seeds=[0,1,2,3,4] | two-sided p=4.999750012499375e-05; mean_left_minus_right=33892.76457510114; higher_is_better => sac better | yes: primary reward claim only; feasibility audit remains separate |

## Metric-Consistency Notes

### TSO

- The hypothesis tests use `total_operating_cost` on the IID split only. They are cost-ordering tests, not safety-gate tests.
- Hard safety eligibility is separate: `reserve_shortfall_rate == 0.0` and `thermal_violation_rate == 0.0`.
- No IID row in `latest.json` passes both gates: `all_on` has `thermal_violation_rate_mean=0.03458333333333333`; `merit_order` has `reserve_shortfall_rate_mean=0.009583333333333333` and `thermal_violation_rate_mean=0.12416666666666668`; `ppo` has `reserve_shortfall_rate_mean=0.07283333333333333` and `thermal_violation_rate_mean=0.3554166666666667`; `ppo_lagrangian` has `thermal_violation_rate_mean=0.050416666666666665`.
- The appendix should describe TSO as a cost-safety frontier or hard negative result, not as a safety-eligible learned-policy win.

### DSO

- The primary metric is `total_loss_mwh` on IID, lower is better. Voltage violation metrics are physical diagnostics alongside the primary loss metric.
- All four DSO p-values are `0.0625` because the implemented tests are paired seed-level sign-flip tests with only five seed pairs.
- With a two-sided exact sign-flip test at `n=5`, the p-value floor prevents a `p<0.05` statement even when all paired differences have the same sign.
- `NormScore` is not the paper-facing claim basis for DSO because `latest.json` marks `norm_score_status=unstable_anchor_gap`; use raw `total_loss_mwh` and voltage metrics.

### DERs

- The primary hypothesis-test metric is IID `mean_p_loss_mw`, lower is better. Do not replace it with reward or `voltage_tightening` metrics.
- `voltage_tightening` is stress evidence, not the primary hypothesis test.
- In `latest.json`, `voltage_tightening` `voltage_violation_steps_mean` is `ippo=4.913333333333333`, `ippo_safe=4.906666666666667`, `no_control=8.733333333333333`, `volt_droop=5.966666666666667`, and `ippo_lagrangian=7.9399999999999995`.
- IPPO and IPPO-rs significantly reduce IID `mean_p_loss_mw` versus `no_control` and `volt_droop`. IPPO-Lagrangian is not significant versus `no_control` and is significantly worse than `volt_droop` on the primary loss metric.

### GenCos

- The primary metric is `total_profit` on IID, higher is better.
- IPPO significantly beats `truthful` and `uniform_mid`, but is significantly below `max_markup`.
- This supports a market benchmark-hardness interpretation rather than a claim that learning dominates the strongest heuristic.
- `sced_convergence_rate` is a physical/market validity diagnostic in the rows, not the hypothesis-test target.

### DC Microgrid

- The hypothesis tests use IID `episode_reward`, higher is better. They do not test feasibility.
- Feasibility and physical consistency remain separate audit channels, including `feasibility_rate`, `mean_cost_power_balance`, and `sla_violation_rate` fields in `latest.json` plus the dedicated paper/physical audit artifacts.
- Do not use `episode_reward` p-values as evidence that strict feasibility passed; report reward ordering and feasibility audit results as separate claims.
