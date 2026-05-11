# PR4 Algorithm v3 Mechanism Positioning

ACD/NISP:
ACD and NISP remain validated on real timed mini-SWE traces. ACD accounts only for prefix-safe context reuse; NISP accounts for measured tool-stall state parking and resume prefill reduction.

TAPS-v3:
TAPS-v3 is the main scheduling/runtime contribution when validation shows p95 JCT or ready-queue wait improvement under multi-session pressure. The main result uses predictive tool latency from leave-one-instance-out command-class medians plus already observed same-session history. taps_oracle_upper_bound is reported only as an upper bound.

CDF:
CDF uses strict/block-prefix accounting. It is a secondary prompt-rendering compiler in the current mini-SWE data because the observed gain is weak; it is not used as the main performance claim unless new data shows stronger non-oracle gain.

PABB-v3:
PABB-v3 is an event-level online branch-budget mechanism. Its main result may use only signals revealed by executed LLM/tool/verifier/patch events. pabb_oracle_upper_bound is an upper bound and cannot be reported as online performance.

Old BES:
Deprecated. It is not restored as a main mechanism and is not used for real mini-SWE main-result attribution.

Correctness:
No solved-rate claims are made without official verifier results. Unknown verifier outcomes remain unknown.
