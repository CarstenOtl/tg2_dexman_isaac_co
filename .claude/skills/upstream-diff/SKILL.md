---
name: upstream-diff
description: Fetch the upstream remote and show a categorized diff — new files only in upstream, modified files, and deleted files — grouped by directory. Use when the user wants to check what's changed in upstream or see if there are new features to pull in.
tools: Bash
---

# Upstream Diff

Fetch upstream and show a structured summary of what's changed.

## Workflow

### Step 1: Check upstream exists

```bash
git remote -v | grep upstream
```

If no upstream remote, report the remotes that exist and stop.

### Step 2: Fetch upstream

```bash
git fetch upstream 2>&1
```

Identify the upstream default branch (check `upstream/main`, `upstream/master`).

### Step 3: Get file-level diff

```bash
git diff HEAD upstream/main --name-status
```

### Step 4: Categorize results

Group into:

**New files in upstream (not in your fork):**
Lines starting with `A` — files upstream has that you don't.

**Modified files (exist in both, differ):**
Lines starting with `M`.

**Deleted in upstream (you still have them):**
Lines starting with `D`.

### Step 5: Print grouped summary

```
## New files in upstream
dextrah_lab/distillation_new/eval_student.py
dextrah_lab/distillation_new/eval_utils.py
...

## Modified files (N total)
### dextrah_lab/tasks/
  dextrah_lab/tasks/tg2_inspirehand/dextrah_tg2_inspirehand_env.py
### dextrah_lab/distillation/
  ...

## Deleted in upstream
  ...

## Recent upstream commits (not in your branch)
<git log upstream/main ^HEAD --oneline -10>
```

At the end, suggest which new files look worth copying in (e.g. new eval scripts, new task variants) and flag any modified core files that might have fixes or features worth reviewing.
