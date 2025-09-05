---
description: Make appropriate commits with concise messages and push changes to GitHub
---

I'll analyze the current changes, create appropriate commits with concise messages, and push to GitHub.

First, let me check the current status:

!git status

Let me check what changes exist:

!git diff --name-only
!git diff --cached --name-only

Let me review recent commit style:

!git log --oneline -3

Now I'll stage the changes and create a commit. Let me check if there are untracked files:

!git ls-files --others --exclude-standard

I'll stage all relevant changes:

!git add .

Let me create a concise commit message based on the staged changes:

!git diff --cached --stat

!git commit -m "$(cat <<'EOF'
$(git diff --cached --name-only | head -3 | sed 's/.*\///' | sed 's/\.[^.]*$//' | tr '\n' ', ' | sed 's/, $//' | sed 's/^/Update /')

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

Finally, push to GitHub:

!git push

Changes have been committed and pushed with a concise, descriptive message.