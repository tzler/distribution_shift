#!/bin/bash
# Installs the pre-commit hook that lints any staged state file. Run once per clone.
cd "$(git rev-parse --show-toplevel)" && cat > .git/hooks/pre-commit <<'H'
#!/bin/bash
files=$(git diff --cached --name-only | grep -E '^(states/0.*\.md|STATE\.md)$')
[ -z "$files" ] && exit 0
python scripts/state_lint.py $files || { echo; echo "state_lint failed: fix the hits or silence a defined term with <!-- lint: ok term -->"; exit 1; }
H
chmod +x .git/hooks/pre-commit && echo "pre-commit hook installed"
