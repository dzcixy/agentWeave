# PR4 Algorithm v2 Mechanism Positioning

CDF:
CDF now uses strict-prefix accounting. Natural reuse is counted only as a continuous prefix from token/block position 0. Because raw token ids are unavailable in the mini-SWE traces, PR4-v2 uses block-prefix mode over ordered context segment token hashes and lengths. Segment reuse away from position 0 is reported as potential only and is not counted as KV-safe without CDF canonical rendering.

TAPS:
The main TAPS result uses predictive tool latency. The predictor is trained from observed tool events with leave-one-instance-out medians by tool type and command class, plus previous latency from the same session when already observed. taps_oracle is retained only as an upper bound and is not reported as the online algorithm.

PABB:
The main PABB result uses online signals only. patch_candidate_seen, duplicate_patch_seen_so_far, pytest_seen, return codes, and verifier status become visible only after the corresponding trace prefix has executed. pabb_oracle_upper_bound is retained only as an upper bound.

Old BES:
Deprecated. It is not restored as a main mechanism and is not used for real mini-SWE main-result attribution.

Real mini-SWE attribution:
Only non-oracle evidence should be used for real mini-SWE gains. Unknown verifier results remain unknown and no solved rate is reported without official harness evaluation.
