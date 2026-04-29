# Local Inference Stability Research Notes

Date: 2026-04-29

This note tracks primary references reviewed for improving browser-local LLM reliability on low-memory devices, attachment-heavy prompts, and WebGPU execution stability.

## Core WebLLM / MLC references
1. https://github.com/mlc-ai/web-llm
2. https://webllm.mlc.ai/docs/user/basic_usage.html
3. https://webllm.mlc.ai/docs/user/api_reference.html
4. https://llm.mlc.ai/docs/deploy/webllm.html
5. https://llm.mlc.ai/docs/deploy/mlc_chat_config.html
6. https://llm.mlc.ai/docs/get_started/quick_start
7. https://github.com/mlc-ai/mlc-llm

## WebGPU platform references
8. https://www.w3.org/TR/webgpu/
9. https://gpuweb.github.io/gpuweb/
10. https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
11. https://developer.mozilla.org/en-US/docs/Web/API/GPUAdapter
12. https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice
13. https://chromestatus.com/feature/6213121689518080

## Runtime capability signals
14. https://developer.mozilla.org/en-US/docs/Web/API/Navigator/deviceMemory
15. https://developer.mozilla.org/en-US/docs/Web/API/Navigator/hardwareConcurrency
16. https://developer.mozilla.org/en-US/docs/Web/API/Navigator/userAgent

## PDF extraction / rendering references
17. https://mozilla.github.io/pdf.js/
18. https://github.com/mozilla/pdf.js
19. https://mozilla.github.io/pdf.js/examples/

## Performance / memory practical references
20. https://web.dev/articles/webgpu-best-practices
21. https://web.dev/articles/off-main-thread
22. https://web.dev/articles/memory-performance
23. https://developer.chrome.com/docs/devtools/memory-problems
24. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_Management

## Model/tokenization + context pressure references
25. https://huggingface.co/docs/transformers/main/en/llm_tutorial
26. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one
27. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_many
28. https://arxiv.org/abs/2309.06180
29. https://arxiv.org/abs/2307.08691

## Browser storage / persistence references
30. https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
31. https://developer.mozilla.org/en-US/docs/Web/API/Cache
32. https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API

## Key conclusions applied in codebase
- Keep attachment payloads out of visible chat messages and avoid inline data URLs/base64 for stability and UX.
- Keep PDF payloads in extracted-document channel; do not duplicate full content in user message.
- Prefer conservative runtime context + prefill on low-memory devices.
- Reduce auxiliary planning/reviewer passes on constrained runtimes.
- Ensure clear/reset fully clears persisted context and in-memory state.
