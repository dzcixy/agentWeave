# Final Algorithm Logic PR4-v7

## Problem
Agent-on-wafer serving is a stateful execution-graph problem, not a flat request-stream scheduling problem. Coding agents repeatedly reuse task/repo/tool context, fork branch-local work, block on shell tools, and later resume with discontinuous state.

## Challenges
- Context-domain fragmentation: identical task/repo/tool context can be placed and prefetched repeatedly when the runtime ignores the agent graph.
- Tool-stall state discontinuity: tool calls leave compute regions idle while private branch state may be evicted or recomputed before resume.
- Regime-dependent multi-session scheduling: admission-heavy, domain-hot, tail-risk, and memory-pressure regimes favor different non-oracle schedulers.

## Algorithms
- Agent Execution Graph: converts trajectories into LLM/tool/verifier DAGs with context segments and branch structure.
- ACD: places shared context domains in a wafer-resident arena to reduce repeated prefill and remote KV movement.
- NISP: parks HOT/WARM/COLD branch state across tool stalls to reduce resume prefill.
- TAPS-P: a regret-aware policy portfolio that selects among non-oracle scheduling policies using train-derived, online-visible configuration/regime features.

## Demoted Mechanisms
- CDF is optional because strict-prefix mini-SWE gains are weak in the current traces.
- PABB-S is optional until real patch snapshot events are available.
- Old BES is deprecated and is not used as a real mini-SWE main-result mechanism.

## Evidence Boundaries
- ACD/NISP evidence is token and model-side replay evidence on real mini-SWE traces.
- TAPS-P targets multi-session p95/throughput under fair strongest-baseline comparison.
- No solved-rate claim is made unless official verifier coverage is available.
