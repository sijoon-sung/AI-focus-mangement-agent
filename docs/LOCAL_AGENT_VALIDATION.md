# Planwise validation — 2026-10-14

Environment: Windows, Python 3.14.7, Tk 9.0, SQLite 3.53.2.
Runtime dependencies: Python standard library only. GUI inspection used Pillow
from a separate development environment; Pillow is not an app dependency.

Passed locally:

- 13 core tests: dependency restrictions/cycles, safe session switching,
  return notes, wait semantics, exact explanation sum, stored trace/feedback,
  minimum-sample personal duration adjustment, explicit category learning,
  rejection of raw observation fields, retention, deletion and category mapping.
- 3 desktop integration tests: session → pause → explanation → completion;
  waiting timer without falsely completing the task; opt-in and excluded
  categories with mocked sensor input. Real activity collection was not enabled.
- UI smoke: three tabs, example tasks, recommendation and explanation rendering.
- Local GUI visual inspection and DPI/layout adjustment.
- Python compilation and Git whitespace checks.

Commands:

```powershell
$env:PLANWISE_UI_TESTS='1'
python -m unittest discover -s tests -p 'test_local_agent*.py' -v
python -m local_agent --demo --ui-smoke --data-dir runs/ui-smoke
python -m compileall -q local_agent
git diff --check
```

This sandbox required TEMP/TMP to point to an ignored directory under runs/
for temporary test files. Normal installations can use the default temp folder.

Not yet validated: real foreground sampling across protected/elevated apps,
long-running sleep/resume, other operating systems, hardware resource budgets,
user productivity effects, Windows packaging, or hosted CI results. The app has
no trained cognitive-state model and does not infer concentration from idle time.

Known limits: app-window notifications only; wait timers are reminders, not
process-completion detection; no deadlines/calendar/day assignment; unencrypted
local SQLite; session time includes unpaused breaks. See the project plan for
the staged implementation and evaluation scope.
