from __future__ import annotations

import json
from pathlib import Path
import logging
import os
from typing import List, Dict

from dotenv import load_dotenv
from .organization.organization import build_organigram_xml
from .preprocess import preprocess_legal_text

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

def _load_few_shot_examples(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(text: str, out_dir: Path) -> None:
    try:
        # Load .env and get Gemini API key
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Please define it in your .env file."
            )

        # 1) Preprocess legal text
        preprocessed_text, _references = preprocess_legal_text(
            input_text=text,
            path=out_dir,
        )
        # preprocessed_text is a string, _references is a list (we ignore it here)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 2) Dummy process model output (your existing placeholder)
        (out_dir / "process.txt").write_text(
            "Controller notifies supervisory authority within 72 hours\n"
            "Processor notifies controller without undue delay\n",
            encoding="utf-8",
        )

        # 3) Extract organization model using LLM (Gemini)
        _xml = build_organigram_xml(
            preprocessed_text= preprocessed_text,
            api_key=api_key,
            output_dir=out_dir,
            model="gemini-2.5-flash",
        )

        (out_dir / "role_task_mapping.xml").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<roleTaskMapping>
  <task id="t1" modality="obligation">
    <roleRef>r1</roleRef>
    <action>notify personal data breach</action>
  </task>
</roleTaskMapping>
""",
            encoding="utf-8",
        )

    except Exception as e:
        logger.exception("Pipeline FAILED")
        raise
