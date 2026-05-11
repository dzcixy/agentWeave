# Metric Dictionary PR4-v14

prefill_compute_tokens = uncached input tokens charged to prefill compute after ACD/NISP cache hits.
resume_prefill_tokens = tokens recomputed on LLM resume; equals uncached prefill work in replay.
cache_hit_tokens = shared_context_hit_tokens + private_suffix_hit_tokens.
shared_context_hit_tokens = immutable shared segment tokens served by ACD residency.
private_suffix_hit_tokens = branch-private suffix tokens preserved by NISP parking.
local_context_bytes = cache/resident bytes read locally on wafer or HBM.
remote_context_bytes = payload context bytes that must move remotely before hop weighting.
remote_kv_bytes = charged NoC bytes used by the simulator, including hop/placement effects.
schedule_remote_kv_bytes = sum of remote_kv_bytes in the replay schedule JSONL(s) at the row's aggregation grain.
astra_policy_remote_bytes = communication bytes emitted by ASTRA exporter; must match the exact schedule JSONL exported.
model_side_latency = prefill_latency + decode_latency + local_memory_latency + noC_latency + queueing/state-prefetch latency.
noC_latency = remote transfer latency plus hop and contention penalty.
tool_latency = observed trace tool latency, reduced only for explicitly enabled safe STP.
end_to_end_jct = model_side_latency + tool_latency on the replayed critical path.
