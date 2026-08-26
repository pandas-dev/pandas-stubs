#!/usr/bin/env bash
set -euo pipefail

CHECK_MODE=0
FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_MODE=1
      shift
      ;;
    --fix)
      CHECK_MODE=0
      shift
      ;;
    *)
      FILES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    while IFS= read -r f; do
      [[ -n "$f" ]] && FILES+=("$f")
    done < <(git ls-files "*.py" "*.pyi")
  fi
fi

VIOLATIONS=0
MODIFIED_FILES=0

for file in "${FILES[@]}"; do
  [[ ! -f "$file" ]] && continue

  TEMP_FILE=$(mktemp)

  awk '
  {
    line = $0
    first_hash = index(line, "#")
    if (first_hash > 0) {
      code_part = substr(line, 1, first_hash - 1)
      comment_part = substr(line, first_hash)
      gsub(/\][ \t]{2,}#/, "] #", comment_part)
      gsub(/\]#[^#]/, "] #&", comment_part)
      gsub(/\] ##/, "] #", comment_part)
      print code_part comment_part
    } else {
      print line
    }
  }
  ' "$file" > "$TEMP_FILE"

  if ! cmp -s "$file" "$TEMP_FILE"; then
    if [[ $CHECK_MODE -eq 1 ]]; then
      echo "Spacing violation in: $file"
      VIOLATIONS=$((VIOLATIONS + 1))
    else
      mv "$TEMP_FILE" "$file"
      echo "Formatted: $file"
      MODIFIED_FILES=$((MODIFIED_FILES + 1))
    fi
  fi
  rm -f "$TEMP_FILE"
done

if [[ $CHECK_MODE -eq 1 && $VIOLATIONS -gt 0 ]]; then
  echo "Found $VIOLATIONS file(s) with comment spacing violations."
  exit 1
elif [[ $CHECK_MODE -eq 0 && $MODIFIED_FILES -gt 0 ]]; then
  echo "Reformatted $MODIFIED_FILES file(s)."
  exit 0
fi

exit 0
