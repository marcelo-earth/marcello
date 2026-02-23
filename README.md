<p align="center">
  <img
    src=".github/logo.png"
    align="center"
    width="100"
    alt="MarceLLo"
    title="MarceLLo"
  />
  <h1 align="center">MarceLLo</h1>
</p>

<p align="center">
  Teaching an LLM to write like Marcelo — using GRPO, not just fine-tuning.
</p>

## Concept

Standard fine-tuning (SFT) memorizes examples. MarceLLo uses **GRPO** (Group Relative Policy Optimization) to let the model *discover* writing style through reinforcement learning, guided by a style classifier as reward signal.

Same technique DeepSeek used for R1 — but the reward is "how much does this sound like Marcelo" instead of "is this reasoning correct."
