# Context Domain Mapping PR4-v13

CONTEXT_DOMAIN_MAPPING_ROWS = 638
CONTEXT_DOMAIN_MAPPING_STATUS = PASS
TOP_SEGMENT_TYPES = [{"actual_cache_hit_tokens": 71314033.69402875, "actual_share": 0.9648343741779295, "predicted_saved_prefill_tokens": 1536766.0, "segment_tokens": 120709, "segment_type": "repo"}, {"actual_cache_hit_tokens": 2599205.305971269, "actual_share": 0.0351656258220705, "predicted_saved_prefill_tokens": 56011.0, "segment_tokens": 2090, "segment_type": "system"}, {"actual_cache_hit_tokens": 0.0, "actual_share": 0.0, "predicted_saved_prefill_tokens": 0.0, "segment_tokens": 49169, "segment_type": "history"}, {"actual_cache_hit_tokens": 0.0, "actual_share": 0.0, "predicted_saved_prefill_tokens": 0.0, "segment_tokens": 2529, "segment_type": "patch"}, {"actual_cache_hit_tokens": 0.0, "actual_share": 0.0, "predicted_saved_prefill_tokens": 0.0, "segment_tokens": 89, "segment_type": "test_log"}]
REPLICATED_SEGMENTS = 151

## Algorithm
Context segments are hashed by segment type, token hash/path, repo and task context. Shared immutable segments create hyperedges from a segment to all consuming LLM events. Placement scores favor saved prefill tokens, penalize remote hop bytes, memory bytes, and hotspot pressure. Replication is allowed only when fanout is high and saved remote-hop cost exceeds memory cost.

## Interpretation
The table links ACD to wafer locality through placement_region, predicted_remote_bytes_saved, actual_cache_hit_tokens, local bytes and remote bytes in the schedule summaries.
