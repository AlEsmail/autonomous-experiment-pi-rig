# ABOUTME: Entry point for the scientist systemd service.
# ABOUTME: Loads config, builds the HTTP client and Anthropic adapter, and starts the poll loop.

import logging
import os
from pathlib import Path

import httpx

from scientist.adapter import AnthropicAdapter
from scientist.loop import Scientist
from scientist.settings import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("scientist")

    repo_root = Path(__file__).resolve().parents[2]
    default_config = repo_root / "config" / "scientist.json"
    config_path = os.environ.get("SCIENTIST_CONFIG", str(default_config))

    cfg = load_config(config_path)
    logger.info("Loaded scientist config from %s", config_path)

    with httpx.Client(base_url=cfg.base_url) as client:
        adapter = AnthropicAdapter(max_tokens=cfg.llm_max_tokens)
        scientist = Scientist(client=client, adapter=adapter, cfg=cfg)
        logger.info("Starting scientist loop against %s", cfg.base_url)
        scientist.run()


if __name__ == "__main__":
    main()
