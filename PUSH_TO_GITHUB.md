# How to publish this on GitHub

This folder is already a git repository with an initial commit. To put it on GitHub:

## Option A — GitHub website (easiest)
1. Go to https://github.com/new and create an empty repository
   (do NOT add a README, .gitignore, or license — this repo already has them).
   Name it e.g. `availability-aware-ids`.
2. In a terminal, from inside this folder:

   ```bash
   git remote add origin https://github.com/<your-username>/availability-aware-ids.git
   git branch -M main
   git push -u origin main
   ```

## Option B — GitHub CLI
```bash
gh repo create availability-aware-ids --public --source=. --remote=origin --push
```

## Notes
- Large raw data files (`*.parquet`, `*.pkl`) are intentionally excluded by `.gitignore`;
  see `data/DATA.md` for how to obtain or regenerate them.
- The initial commit was made with a placeholder author identity. Set your own before
  pushing if you like:
  ```bash
  git config user.name  "Your Name"
  git config user.email "you@example.com"
  git commit --amend --reset-author -m "Initial commit"
  ```
- If your venue requires anonymity for review, create the repo as **private** for now,
  or use an anonymized mirror (e.g. anonymous.4open.science) and add the public link
  only in the camera-ready.
