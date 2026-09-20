# Progress presentations

## 21 September 2026

- [Editable slides](2026-09-21-progress/saert-progress.pptx)
- [Speaker notes](2026-09-21-progress/speaker-notes.md)
- [Meeting guide and Q&A](2026-09-21-progress/meeting-guide.md)

Use slides 1–14 for 12–15 minutes; slides 15–17 are backup. Opening the deck does
not require Python, the corpora or a GPU. The deck was rendered and visually
reviewed; check it in the application you will present with.

To rebuild the same dated deck, run `python -s scripts/prepare_progress_meeting.py`
to prepare inputs from the portable `reports/` bundle. The builder
`scripts/build_progress_presentation.mjs` uses the Codex bundled presentation
runtime. Set `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON` and `SKILL_DIR` to its local
runtime/skill paths. Set `PRESENTATION_NAME` to a new filename and run the builder
with the bundled Node executable. It refuses to overwrite an existing final file.

The deck has editable tables and a native chart. Chart data use two decimal
places for display; report CSVs retain full precision. Private previews and
validation receipts stay under the ignored `.presentation_build/` directory.
