# Changelog

## v0.1.0 — Divide Inbox

- Add a lightweight inbox classifier trained on English messages, with a separate Turkish checkpoint.
- Add the C11 word-level trainer and numeric inference engine using binary context compression.
- Add an English JSON interface with language selection and review handling for unknown vocabulary.
- Include training corpora, held-out template tests, manual diagnostics, and the Turkish dense-attention comparison.
- Preserve the experimental conversational checkpoint separately from the inbox classifier.
- Document the restricted task vocabulary, lossy memory, evaluation overlap, and uncalibrated predictions.
