# Publishing Paper Notes

Upload any Markdown file (`.md`) into this `research-blog/papers/` directory. **Do not edit `index.html`.** The index automatically lists all Markdown files in this directory, and GitHub Pages (Jekyll) creates a readable web page for each one.

For example, upload `repa.md` and the generated article link will be:

https://mm77a.github.io/research-blog/papers/repa/

The article will also appear on https://mm77a.github.io/research-blog/papers/ after GitHub Pages finishes building.

You can use a normal Markdown file **without YAML front matter**. For a prettier title or description, optionally add this to the top:

```yaml
---
title: Understanding REPA
description: My reading notes and open questions
date: 2026-11-22
---
```

If no title is provided, the displayed title comes from the filename. Avoid spaces or non-ASCII symbols in filenames for simpler URLs: prefer `repa.md`, `aceit-notes.md`.

Only publish notes you want publicly accessible; this repository is public. Keep drafts and personal study journals in Obsidian unless you explicitly want to publish them.

**For maintainers:** GitHub Pages should use the `main` branch and `/(root)` and must run Jekyll; do not add a `.nojekyll` file.
