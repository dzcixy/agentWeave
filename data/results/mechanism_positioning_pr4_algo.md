# PR4 Algorithm Mechanism Positioning

ACD/CDF:
Main context-domain mechanism. ACD exploits naturally occurring exact-prefix shared context in the trace, while CDF is a replay-level canonical prompt rendering potential that factors invariant task/repo/tool context into stable context domains. The target is repeated prefill and shared repo-history context.

NISP/TAPS:
Main tool-stall runtime mechanism. NISP handles per-branch state parking across tool stalls. TAPS handles multi-session stall hiding by releasing compute regions while tools run and scheduling ready LLM work using locality, resume urgency, criticality, and age.

PABB:
Branch-budget mechanism. PABB replaces old BES for real coding-agent workloads. It uses patch, test, tool, duplicate-patch, token-cost, and optional official verifier signals. When verifier results are unknown, PABB reports patch/progress metrics only, not solved rate.

Old BES:
Deprecated or folded into PABB. It is not used as the real mini-SWE main-result mechanism because PR3-v4 and PR3-v3 did not observe independent BES gain on timed mini-SWE traces.

Real mini-SWE main result:
Attribute real-trace gains to ACD/CDF/NISP/TAPS and report PABB only as branch-budget/progress control. Do not attribute real mini-SWE gains to old BES.
