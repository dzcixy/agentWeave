# CDF Limitations for PR4-algo-v3

raw_token_ids_available = false
block_prefix_mode = true
natural_strict_prefix_reusable_tokens = 1390910
cdf_canonical_prefix_reusable_tokens = 1394852
cdf_added_reusable_tokens = 3942
segment_reuse_potential_tokens = 1445113

The current mini-SWE traces do not include raw token ids, so strict accounting uses ordered segment token hashes and lengths in block-prefix mode. This is a conservative replay approximation for prefix continuity, not a claim of semantic KV reuse.

The collected traces already have high natural strict/block-prefix reuse because mini-SWE repeatedly renders task, tool, and repository context in a stable order. CDF therefore adds only a small amount of reusable context and is not the main real mini-SWE speedup source in PR4-algo-v3.

CDF remains useful as a prompt-rendering compiler for agent workloads with unstable prompt order or fragmented shared history. In this report it is kept as a secondary mechanism unless stronger non-oracle evidence appears.
