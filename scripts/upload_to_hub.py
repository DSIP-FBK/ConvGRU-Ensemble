#!/usr/bin/env python
"""Upload a trained ConvGRU-Ensemble model to HuggingFace Hub."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g., it4lia/irene)")
    parser.add_argument("--model-card", default=None, help="Path to model card markdown file")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    # Default model card
    model_card = args.model_card
    if model_card is None:
        default_card = Path(__file__).parent.parent / "MODEL_CARD.md"
        if default_card.exists():
            model_card = str(default_card)

    from convgru_ensemble.hub import push_to_hub

    url = push_to_hub(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        model_card_path=model_card,
        private=args.private,
    )
    print(f"Model uploaded: {url}")


if __name__ == "__main__":
    main()
