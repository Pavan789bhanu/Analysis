# GitHub Profile Dashboard — Setup Guide

This folder contains your **GitHub Profile README** — a special README that appears on your GitHub profile page at [github.com/Pavan789bhanu](https://github.com/Pavan789bhanu).

> **Note:** Your `analysis` repo was not modified. This is a standalone deliverable.

---

## Quick Setup (2 minutes)

### Step 1 — Create the profile repository

GitHub profile READMEs live in a repo **with the same name as your username**.

1. Go to [github.com/new](https://github.com/new)
2. Set **Repository name** to exactly: `Pavan789bhanu`
3. Set visibility to **Public**
4. Check **Add a README file** (optional — you'll replace it)
5. Click **Create repository**

### Step 2 — Upload the README

**Option A — Web UI (easiest)**

1. Open your new repo: `github.com/Pavan789bhanu/Pavan789bhanu`
2. Click **Add file → Upload files**
3. Upload `README.md` from this folder
4. Commit directly to `main`

**Option B — Command line**

```bash
git clone https://github.com/Pavan789bhanu/Pavan789bhanu.git
cd Pavan789bhanu
cp /path/to/github-profile-dashboard/README.md ./README.md
git add README.md
git commit -m "Add GitHub profile README dashboard"
git push origin main
```

### Step 3 — Verify

Visit [github.com/Pavan789bhanu](https://github.com/Pavan789bhanu) — your dashboard should appear at the top of your profile within a few seconds.

---

## What's Included

| Section | Description |
|---------|-------------|
| **Hero header** | Animated typing banner with AML + LLM focus |
| **GitHub stats** | Dynamic cards — contributions, streak, top languages |
| **Featured projects** | 6 pinned-style project cards from your recent repos |
| **Professional highlights** | AML, compliance, credit risk, fraud detection metrics from Citigroup & TCS |
| **Research** | Interspeech 2025 & ICMI 2024 publications |
| **Tech stack** | Badge grid covering ML, LLMs, MLOps, and financial domain tools |
| **Experience tree** | ASCII timeline of Citigroup & TCS roles |
| **Activity graph** | Live contribution heatmap |

---

## Customization Tips

- **LinkedIn URL**: Update the badge link in the header if your LinkedIn handle differs
- **Portfolio URL**: Currently points to `pavan-portfolio.com`
- **Pinned repos**: On your GitHub profile, pin these 6 repos for maximum impact:
  1. `recruiter-voice-agent`
  2. `UI_State_Capture_System`
  3. `Pavan-Portfolio`
  4. `Quantization-of-LLMs`
  5. `Auto-Analyst`
  6. `Brain-Tumor-Segmentation`
- **Profile bio**: Go to GitHub Settings → Profile and add:
  > AI/ML Engineer · AML & Financial Crime · LLM Research · M.S. Data Science @ CU Boulder

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| README doesn't show on profile | Repo must be **public** and named exactly `Pavan789bhanu` |
| Stats cards show "error" | GitHub stats API can be slow on first load — refresh after a minute |
| Typing animation not working | The `readme-typing-svg` service may be temporarily down — it auto-recovers |
