# NOESIS-Σ inference examples

These files are deterministic inference and interface fixtures. They are not a
claim of model accuracy by themselves; the expected results in
`production_eval.jsonl` are the acceptance criteria for an evaluation runner.

## Files

- `example.txt`: plain-text and tagged multimodal prompts.
- `example_harmony.json`: four OpenAI-style cases covering vision, structured
  extraction, tool calling, and mid-conversation instruction updates.
- `long_context_prompt.txt`: a text-only prompt for the 262,144-token profile
  that leaves room for a 64K to 128K token response.
- `production_eval.jsonl`: eleven evaluation cases with explicit categories and
  expected-result contracts.
- `validate_examples.py`: dependency-free integrity check. It does not load a
  model or train anything.
- `images/`: valid local JPEG fixtures used by the vision cases.

## Checks

From the project root:

```bash
python examples/validate_examples.py
python -m pytest -q core/encoding/test_encoding.py --tb=short
```

The validator rejects missing files, unhydrated Git-LFS pointer files, invalid
JSON, missing expected results, and missing image references.
