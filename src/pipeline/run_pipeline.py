from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .organigram.api import build_organigram_xml
from src.pipeline.preprocess.api import preprocess_legal_text
from .role_task_mapping.api import build_role_task_mapping

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def run_pipeline(text: str, out_dir: Path) -> None:
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("MODEL")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Please define it in your .env file."
            )

        preprocessed_text, _references = preprocess_legal_text(
            reg_text=text,
            path=out_dir,
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        organigram = build_organigram_xml(
            preprocessed_text=preprocessed_text,
            api_key=api_key,
            output_dir=out_dir,
            model=model,
        )

        build_role_task_mapping(
            preprocessed_text=preprocessed_text,
            organigram = organigram,
            api_key=api_key,
            output_dir=out_dir,
            model=model,
        )

        (out_dir / "process_description.txt").write_text(
            "Controller notifies supervisory authority within 72 hours\n"
            "Processor notifies controller without undue delay\n",
            encoding="utf-8",
        )

    except Exception as e:
        logger.exception("Pipeline FAILED")
        raise
