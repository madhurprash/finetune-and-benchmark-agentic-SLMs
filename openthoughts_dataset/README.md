---
license: apache-2.0
task_categories:
- text-generation
- reinforcement-learning
tags:
- agents
- terminal
- code
- benchmark
library_name: datasets
---

<p align="center">
    <img src="https://huggingface.co/datasets/open-thoughts/OpenThoughts1-Agent-SFT/resolve/main/ota-logo.png" width="50%">
</p>

<p align="center">
<a href="https://www.openthoughts.ai/blog/agent" style="margin-right: 24px;">Project</a> |
<a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT" style="margin-right: 24px; margin-left: 24px;">SFT dataset</a> |
<a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL" style="margin-right: 24px; margin-left: 24px;">RL dataset</a> |
<a href="https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT" style="margin-left: 24px;">SFT model</a>
<a href="https://huggingface.co/open-thoughts/OpenThinker-Agent-v1" style="margin-left: 24px;">RL model</a>
</p>

# OpenThoughts-TB-Dev

## Dataset Description

**OpenThoughts-TB-Dev** is our development benchmark for evaluating agent models on terminal and shell-based tasks. This dataset was curated to measure the effectiveness of different data sources, teacher models, and curation approaches during the development of [OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1). We used this benchmark to ablate over many different instruction generation strategies and evaluate the impact of different teacher models on downstream agent performance. The dataset consists of diverse terminal tasks that require agents to understand instructions, execute shell commands, and interact with file systems and development environments. Performance on OpenThoughts-TB-Dev correlates strongly with performance on larger benchmarks like Terminal-Bench 2.0, making it an efficient evaluation tool for rapid iteration during agent development.

# Links
- 🌐 [OpenThoughts-Agent Project Page](https://www.openthoughts.ai/blog/agent)
- 💻 [OpenThoughts-Agent GitHub Repository](https://github.com/open-thoughts/OpenThoughts-Agent)
- 🧠 [OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- 🧠 [OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL)
- 🤖 [OpenThinker-Agent-v1 model](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)


# Evaluate Your Model on OpenThoughts-TB-Dev

To evaluate your model on this dataset using Harbor, install [Harbor](https://github.com/laude-institute/harbor) and follow these steps:

```
curl -L https://raw.githubusercontent.com/open-thoughts/OpenThoughts-Agent/refs/heads/main/eval/tacc/snapshot_download.py  -o snapshot_download.py

chmod +x snapshot_download.py

python snapshot_download.py   open-thoughts/OpenThoughts-TB-dev   --local-dir <YOUR_LOCAL_DIR>

harbor run --dataset <YOUR_LOCAL_DIR> \
   --agent <AGENT_NAME> \
   --model <MODEL_NAME> \
   --n-concurrent 4
```

All LiteLLM model names are accepted.

   
# Citation
```
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  month = Dec,
  title = {{OpenThoughts-Agent}},
  howpublished = {https://open-thoughts.ai/agent},
  year = {2025}
}
```
