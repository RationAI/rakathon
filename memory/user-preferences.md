---
name: user-preferences
description: How Adam Džadoň wants work done - simple code, uv, no CLI args, CSV outputs, standard library calls over hand-rolled logic
metadata:
  type: feedback
---

Preferences stated by Adam Džadoň (`ejdam87`). Other team members may differ.

- **Keep it simple.** He pushed back on over-engineering: rejected a checksum manifest ("just use seed why hashing") and a hand-written shuffle ("why so complicated, why not stratified split"). Prefer a standard library call (sklearn etc.) over custom logic.
- **Scripts take no CLI arguments**; use constants at the top of the file.
- **One uv project for the whole repo** (root `pyproject.toml`); no per-folder pyprojects or venvs. Add dependencies with `uv add` from the root.
- **CSV over Parquet** for data outputs.
- Exploration goes in notebooks with markdown exports. `exploration/playground.ipynb` is his gitignored scratchpad, edited live in JupyterLab (he must reload from disk after an agent edits it).
- Gitignore generated / local files (data, venvs, checkpoints, caches).
- Short, terse requests: act directly, explain briefly. He commits himself.

**Why:** corrections he gave during the session. **How to apply:** default to the simplest standard approach and ask before adding extra machinery. Related: [[split-decision]], [[repo-layout]].
