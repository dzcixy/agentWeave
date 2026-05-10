# vLLM Metrics Source Audit PR2-v2

rows_parsed = 0
PREFIX_CACHE_METRICS_RELIABLE = false

## prefix_cache_queries

- exact_metric_names_allowed = vllm:prefix_cache_queries_total, vllm_prefix_cache_queries_total, vllm:prefix_cache_query_total, vllm_prefix_cache_query_total
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = true
- matched_sources = 0
- used none

## prefix_cache_hits

- exact_metric_names_allowed = vllm:prefix_cache_hits_total, vllm_prefix_cache_hits_total, vllm:prefix_cache_hit_total, vllm_prefix_cache_hit_total
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = true
- matched_sources = 0
- used none

## cached_prompt_tokens

- exact_metric_names_allowed = vllm:cached_prompt_tokens_total, vllm_cached_prompt_tokens_total, vllm:prompt_tokens_cached_total, vllm_prompt_tokens_cached_total, vllm:prefix_cache_hit_tokens_total, vllm_prefix_cache_hit_tokens_total
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = true
- matched_sources = 0
- used none

## prompt_tokens

- exact_metric_names_allowed = vllm:prompt_tokens_total, vllm_prompt_tokens_total, vllm:prompt_tokens, vllm_prompt_tokens
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = false
- matched_sources = 0
- used none

## generation_tokens

- exact_metric_names_allowed = vllm:generation_tokens_total, vllm_generation_tokens_total, vllm:decode_tokens_total, vllm_decode_tokens_total, vllm:output_tokens_total, vllm_output_tokens_total
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = false
- matched_sources = 0
- used none

## num_requests_running

- exact_metric_names_allowed = vllm:num_requests_running, vllm_num_requests_running
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = false
- matched_sources = 0
- used none

## num_requests_waiting

- exact_metric_names_allowed = vllm:num_requests_waiting, vllm_num_requests_waiting
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = false
- matched_sources = 0
- used none

## gpu_kv_cache_usage

- exact_metric_names_allowed = vllm:gpu_cache_usage_perc, vllm_gpu_cache_usage_perc, vllm:gpu_kv_cache_usage_perc, vllm_gpu_kv_cache_usage_perc
- allowed_metric_types = counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = true
- prefix_cache_metric = false
- matched_sources = 0
- used none

## request_queue_time

- exact_metric_names_allowed = vllm:request_queue_time_seconds_sum, vllm_request_queue_time_seconds_sum, vllm:time_in_queue_requests_seconds_sum, vllm_time_in_queue_requests_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

## request_prefill_time

- exact_metric_names_allowed = vllm:request_prefill_time_seconds_sum, vllm_request_prefill_time_seconds_sum, vllm:prefill_time_seconds_sum, vllm_prefill_time_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

## request_decode_time

- exact_metric_names_allowed = vllm:request_decode_time_seconds_sum, vllm_request_decode_time_seconds_sum, vllm:decode_time_seconds_sum, vllm_decode_time_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

## ttft

- exact_metric_names_allowed = vllm:time_to_first_token_seconds_sum, vllm_time_to_first_token_seconds_sum, vllm:request_ttft_seconds_sum, vllm_request_ttft_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

## tpot

- exact_metric_names_allowed = vllm:time_per_output_token_seconds_sum, vllm_time_per_output_token_seconds_sum, vllm:request_tpot_seconds_sum, vllm_request_tpot_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

## e2e_latency

- exact_metric_names_allowed = vllm:e2e_request_latency_seconds_sum, vllm_e2e_request_latency_seconds_sum, vllm:request_latency_seconds_sum, vllm_request_latency_seconds_sum, vllm:request_duration_seconds_sum, vllm_request_duration_seconds_sum
- allowed_metric_types = histogram, summary, counter, gauge, untyped, unknown
- excludes_bucket_sum_count_for_token_counter = false
- prefix_cache_metric = false
- matched_sources = 0
- used none

