# PR4-v5 Mechanism Positioning

- ACD/NISP remain validated on real mini-SWE traces and are retained as the reliable base mechanisms.
- TAPS-U status: STRONG. It is the main algorithm only when it beats `reactive_admission` or `acd_nisp` on p95 JCT or throughput under non-oracle high-pressure serving replay.
- CDF is optional, not a main result, because strict/block-prefix CDF gain remains weak on current mini-SWE traces.
- PABB-S status: NOT_OBSERVED. It is main only if real patch snapshot events improve online branch budgeting; command-visible proxies are not enough.
- Old BES is deprecated and is not used as a real mini-SWE main-result mechanism.
- No solved-rate claim is made without official SWE-bench verifier coverage.
