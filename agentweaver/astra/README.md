# AgentWeaver ASTRA Bridge

This directory contains a skeleton bridge from AgentWeaver traces to an
ASTRA-sim / Chakra-style intermediate JSON trace.

Current status:

- Export format: `intermediate_json`
- Real Chakra protobuf generation: not implemented
- ASTRA-sim execution: external, optional, and never faked

Mapping:

- LLM prefill -> compute node
- LLM decode -> compute node
- KV/context fetch -> memory node
- Remote KV/context movement -> communication node
- Tool call -> delay node
- Verifier -> delay or compute node

Use:

```bash
python -m agentweaver.astra.run_astra_smoke
```

This writes smoke and one real mini-SWE intermediate export under
`data/astra_traces/` and `data/astra_configs/`.

To run a real ASTRA binary, set `ASTRA_SIM_PATH` and optionally
`ASTRA_RUN_CMD`, then use:

```bash
bash scripts/run_astra_smoke.sh
```

If `ASTRA_RUN_CMD` is not provided, the script exports traces and reports that
ASTRA execution was not run rather than fabricating cycles.
