# /// script
# requires-python = ">=3.10"
# dependencies = ["gguf"]
# ///
"""Extract embedded chat templates from GGUF files for inspection and diffing."""

import sys
from pathlib import Path


def extract_template(gguf_path: Path) -> str | None:
    """Extract the tokenizer.chat_template metadata from a GGUF file."""
    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))
    for field in reader.fields.values():
        if field.name == "tokenizer.chat_template":
            return bytes(field.parts[-1]).decode("utf-8")
    return None


def main():
    models_root = Path(__file__).parent / "models"
    out_dir = Path(__file__).parent / "extracted-templates"
    out_dir.mkdir(exist_ok=True)

    if not models_root.exists():
        print(f"Models root not found: {models_root}")
        return 1

    gguf_files = sorted(models_root.rglob("*.gguf"))
    if not gguf_files:
        print("No .gguf files found.")
        return 1

    print(f"Found {len(gguf_files)} GGUF files. Extracting chat templates...\n")

    for gguf_path in gguf_files:
        rel = gguf_path.relative_to(models_root)
        print(f"  {rel} ... ", end="", flush=True)

        try:
            template = extract_template(gguf_path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if template is None:
            print("no chat_template found")
            continue

        # Save with a flat filename: folder__model.jinja
        safe_name = str(rel).replace("\\", "__").replace("/", "__")
        safe_name = safe_name.rsplit(".", 1)[0] + ".jinja"
        out_path = out_dir / safe_name
        out_path.write_text(template, encoding="utf-8")
        print(f"saved ({len(template)} chars)")

    print(f"\nAll templates saved to: {out_dir}")
    print(f"\nTo diff against your local overrides, run:")
    print(f'  diff extracted-templates/<model>.jinja models/chat_template_gemma.jinja')
    print(f'  diff extracted-templates/<model>.jinja models/chat_template.jinja')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
