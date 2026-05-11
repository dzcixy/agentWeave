# AgentWeaver v12 Layered Ablation

FULL_AGENTWEAVER_P95_GAIN_OVER_REACTIVE = 0.064870
TAPS_C_INCREMENTAL_P95_GAIN = 0.064870
STP_AE_INCREMENTAL_P95_GAIN = 0.000000

TAPS-C remains a weak validity-aware compiler in this run; its matched p95 lift is small.
The simulator exposes ACD and NISP as a combined acd_nisp policy here, so isolated ACD-vs-NISP JCT attribution is not claimed.
STP-AE is artifact-equivalent and safe, but currently shows mean-only improvement and no p95 improvement.
