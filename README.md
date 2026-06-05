# awesome-ascend-plugins

A marketplace of Ascend NPU developer plugins (Claude Code plugin format).

## Plugins

| Plugin | Path | Description |
|---|---|---|
| [ascendc-operator-studio](ascend_AscendC_plugins/README.md) | [`ascend_AscendC_plugins/`](ascend_AscendC_plugins/) | Self-contained, closed-loop agent plugin for **end-to-end AscendC custom operator development** (ascend-kernel / `csrc/ops` + `build.sh` + `torch_npu`): env → init → two-level tiling design → code-gen → compile/debug → docs → precision (+debug) → performance → optimization → code review. Bundles the orchestrator agent, a kickoff command, and the full `ascendc` skill (references, templates, scripts, examples). |

## Install

This repository is a plugin marketplace defined by
[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json). Add the marketplace
and install a plugin from it, then use its slash command, agent, or skill.

For `ascendc-operator-studio`:

- Command: `/ascendc-operator <op_name> "<math/functional spec>"`
- Agent: `ascendc-operator-studio`
- See [ascend_AscendC_plugins/README.md](ascend_AscendC_plugins/README.md) for details.
