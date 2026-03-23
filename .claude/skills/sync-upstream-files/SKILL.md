---
name: sync-upstream-files
description: Interactively show new files present in upstream but not in your fork, let the user pick which ones to copy over, then copy them — without doing a full merge. Use when the user wants to pull specific files or features from upstream without merging all changes.
tools: Bash, Read, Write
---

# Sync Upstream Files

Selectively copy new files from upstream into your fork — no merge, no conflicts.

## Workflow

### Step 1: Fetch upstream

```bash
git fetch upstream 2>&1
git remote -v | grep upstream
```

Identify the upstream branch (main/master).

### Step 2: Find new files only in upstream

```bash
git diff HEAD upstream/main --name-status | grep "^A"
```

These are files upstream has that your fork doesn't yet.

### Step 3: Show the list

Present the new files grouped by directory. For each file, show:
- File path
- File size: `git show upstream/main:<path> | wc -c`
- First docstring line (if Python): `git show upstream/main:<path> | head -5`

Example:
```
New files available in upstream:

dextrah_lab/distillation_new/
  eval_student.py  (28KB) — "Evaluate a student checkpoint with SafeDagger runtime eval semantics"
  eval_utils.py    (8KB)  — "Shared evaluation helpers and metric definitions"

dextrah_lab/rl_games/
  eval_teacher.py  (52KB) — "Evaluate teacher policy checkpoints and report lift success/unsafe rate"
```

### Step 4: Ask which to copy

Ask the user: "Which files would you like to copy? (all / list numbers / none)"

### Step 5: Copy selected files

For each selected file:
```bash
git show upstream/main:<path> > <local_path>
```

Confirm each file was written successfully with size.

### Step 6: Summary

List all files copied, note they are untracked (not staged). Remind the user to commit when ready.

Do NOT copy modified files (only new files) — this skill is for adding missing files only, not overwriting local changes.
