# AgentWeaver v12 Regime Analysis

Regime analysis keeps the validation config set fixed and reports where mechanisms help. STP-AE class-level upper bounds are excluded.

- tool-heavy: configs=174 best=TAPS-C selection p95_gain=0.007754 mean_gain=0.019972 throughput_gain=-0.001565
- model-heavy: configs=14 best=reactive/best-fixed p95_gain=0.003039 mean_gain=0.028556 throughput_gain=0.001560
- context-heavy: configs=81 best=reactive/best-fixed p95_gain=0.000204 mean_gain=0.004109 throughput_gain=-0.000394
- high-concurrency: configs=102 best=reactive/best-fixed p95_gain=0.000074 mean_gain=0.004122 throughput_gain=-0.000463
- high-region-pressure: configs=147 best=TAPS-C selection p95_gain=0.014506 mean_gain=0.037634 throughput_gain=-0.002681
- bursty arrival: configs=110 best=reactive/best-fixed p95_gain=0.004823 mean_gain=0.019872 throughput_gain=0.000931
- low-memory: configs=162 best=TAPS-C selection p95_gain=0.006988 mean_gain=0.018859 throughput_gain=-0.002084
- high-memory: configs=79 best=reactive/best-fixed p95_gain=0.007026 mean_gain=0.019669 throughput_gain=-0.000775
