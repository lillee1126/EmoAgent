from __future__ import annotations

import argparse
import json
import os

from .build_embeddings import ArkTextEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Ark credentials and embedding endpoint")
    parser.add_argument("--model", default=os.environ.get("ARK_EMBEDDING_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL"))
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or ARK_EMBEDDING_MODEL is required")
    vector = ArkTextEmbedder(args.model, args.base_url, max_retries=1).embed(
        "emotion factor knowledge base preflight check"
    )
    print(
        json.dumps(
            {
                "ark_preflight": "ok",
                "model": args.model,
                "dimension": int(vector.size),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
