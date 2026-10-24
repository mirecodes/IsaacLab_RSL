# Forge agent guide

Forge holds our own Isaac Lab tasks. Read this before adding or changing a feature here.

## Layout

```
forge/
  template/                    copy this to start a task
  <domain>/                    e.g. velocity/
    <task>/                    e.g. go2_velocity/
      __init__.py              gym.register only
      env_cfg.py               one *_env_cfg.py per variant if needed
      agents/rsl_rl_ppo_cfg.py
      mdp/                     this task's own terms
      REPORT.md
  terrain/                     terrain-generation templates (templates/<type>.py) and play tasks
                               IsaacContrib-Terrain-<Type>-<Policy>; add a terrain type there, see its REPORT.md
```

## Workflow

1. **New task or edit?** Extending an existing task's study → edit that task. Otherwise → `cp -r template <domain>/<task>` (snake_case). Ask if unclear.
2. **Choose the base** in `env_cfg.py`: inherit the closest existing task's env cfg (core, contrib or forge), or build from scratch on `ManagerBasedRLEnvCfg`. Do the same for the runner cfg.
3. **Rename** in the copy: `Template*` classes, gym ID `IsaacContrib-<Domain>-<Task>-<Variant>`, `experiment_name`. Uncomment the registration in `__init__.py`.
4. **Implement as manager terms** (command, observation, action, reward, event, termination, curriculum) set in the env cfg. Subclass the env only when terms cannot express it.
5. **Write `REPORT.md`** (below).
6. **Verify** (below).

## Task independence

- Each task owns its `mdp/`. Never import another task's `mdp/` or configs. To reuse a term, copy it and note the origin in `REPORT.md` (**Copied from**).
- Upstream packages (`isaaclab`, `isaaclab_tasks.core`, `isaaclab_rl`, `isaaclab_assets`) may be imported or inherited freely.
- Inside a task's `mdp/`, group related terms in one file per feature (e.g. `cpg.py` = action term, phase observation, gait rewards). Put robot-specific values (body names, scales) in params, not in module constants.

## `REPORT.md`

A short paper-style note, equations first. Use the sections in the template's `REPORT.md`.

- **Architecture**: the whole control pipeline (command → observation → networks → action → actuator → sim) as a flowchart, with the rate of each module and the shape of every network. Update it whenever a module or the network shape changes.
- **Method**: only what differs from the base. Give each term as an equation, with its symbols, weight and range.
- **Deviations**: values the sources do not fix, each with its reason.
- **Editing a task**: rewrite Architecture and Method so they stay current, and append to the Changelog: `YYYY-MM-DD: change. Reason.`

## Code

- Match `velocity/go2_velocity`: a comment with the formula and source above each weight or range.
- Use the SPDX header from the existing files, and English comments.
- Vectorize over `env_ids`; no per-env Python loops.
- Mid-episode changes: an event with `mode="interval"`. Conditional triggers: inside the event, filter `env_ids` by the condition. Training-progress changes: a curriculum term (see `velocity/go2_velocity/mdp/curriculums.py`).
- Runtime limits: no adding or removing prims, no terrain mesh changes. Pre-spawn objects and move them instead. Under Newton, call `add_model_change` after writing model attributes (see `velocity/go2_velocity/mdp/events.py`).
- Networks: MLP, RNN and CNN cfgs live in `isaaclab_rl/rsl_rl/rl_cfg.py`. A custom `nn.Module` goes in `<task>/agents/`. First check how the installed rsl_rl resolves `class_name`.
- Every package under `forge/` except `mdp/` is imported at startup to run `gym.register`. So `__init__.py` files outside `mdp/` import nothing heavy.

## Verify

- Run the repo's format and lint checks (see the repo's `AGENTS.md`).
- Run a short training on Euler through `euler` (e.g. `--max_iterations 2`). Don't report a task as working before this passes.
