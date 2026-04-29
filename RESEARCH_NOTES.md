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
20. https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html
21. https://tesseract-ocr.github.io/tessdoc/
22. https://cloud.google.com/document-ai/docs/ocr
23. https://learn.microsoft.com/azure/ai-services/document-intelligence/
24. https://docs.aws.amazon.com/textract/latest/dg/what-is.html

## Performance / memory practical references
25. https://web.dev/articles/webgpu-best-practices
26. https://web.dev/articles/off-main-thread
27. https://web.dev/articles/memory-performance
28. https://developer.chrome.com/docs/devtools/memory-problems
29. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_Management

## Model/tokenization + context pressure references
30. https://huggingface.co/docs/transformers/main/en/llm_tutorial
31. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one
32. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_many
33. https://arxiv.org/abs/2309.06180
34. https://arxiv.org/abs/2307.08691

## Browser storage / persistence references
35. https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
36. https://developer.mozilla.org/en-US/docs/Web/API/Cache
37. https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API

## Key conclusions applied in codebase
- Keep attachment payloads out of visible chat messages and avoid inline data URLs/base64 for stability and UX.
- Keep PDF payloads in extracted-document channel; do not duplicate full content in user message.
- Prefer conservative runtime context + prefill on low-memory devices.
- Reduce auxiliary planning/reviewer passes on constrained runtimes.
- Ensure clear/reset fully clears persisted context and in-memory state.
