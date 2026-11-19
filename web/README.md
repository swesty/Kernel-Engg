# Kernel tools website

Live hub: https://pages.swesty.net/kernels/

Run `uv run scripts/build-kernels-site.py` from the repository root to refresh the standalone tools (Qwen3 block, GEMM, shared memory, FlashAttention, SM12x attention, TensorFold vs vLLM) and extract the Mermaid diagram from the assignment source. The SM12x attention map and its report (`guides/sm12x-attention/`), and the TensorFold vs vLLM report (`guides/tensorfold-vs-vllm/report.html`), come from the consumer-gpu-kernel-guide repository; the script expects that checkout at `~/Developer/projects/consumer-gpu-kernel-guide` or at `$CONSUMER_GPU_GUIDE`, and renders `REPORT.md` to `sm12x-attention/report/` with Python-Markdown, which `uv run` installs from the script header. The hub and diagram wrapper are maintained in `web/kernels/`. Only this directory is published; assignment notebooks and results are excluded.

On Mac Studio, Caddy serves `/Users/swesty/Sites/pages` behind the existing HTTPS proxy. Publish only `web/kernels/` to `/Users/swesty/Sites/pages/kernels/`. Do not replace the parent directory: `/brief/` is managed separately by the intel brief automation. No DNS or service changes are needed. Access follows the existing pages.swesty.net network configuration.

The diagram uses a local copy of Mermaid 10.9.3 (MIT), downloaded from https://cdn.jsdelivr.net/npm/mermaid@10.9.3/dist/mermaid.min.js. Its license is in `kernels/assets/MERMAID-LICENSE`.
