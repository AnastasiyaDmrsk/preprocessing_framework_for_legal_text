from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.AutoBPMN_testset.api import build_testset
from src.pipeline.organigram.api import build_organigram_xml
from src.pipeline.preprocess.api import preprocess_legal_text
from src.pipeline.process_description.api import build_process_description
from src.pipeline.role_task_mapping.api import build_role_task_mapping

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

        role_task_mapping = build_role_task_mapping(
            preprocessed_text=preprocessed_text,
            organigram = organigram,
            api_key=api_key,
            output_dir=out_dir,
            model=model,
        )

        process_description = build_process_description(
            preprocessed_text=preprocessed_text,
            role_task_mapping=role_task_mapping,
            api_key=api_key,
            output_dir=out_dir,
            model=model,
        )

        build_testset(
            process_description=process_description,
            role_task_mapping=role_task_mapping,
            api_key=api_key,
            output_dir=out_dir,
            base_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
            job_id=out_dir.name,
            model=model,
            use_validator=True,
            form_url=f"{os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")}/forms/generic.html",
        )

    except Exception as e:
        logger.exception("Pipeline FAILED")
        raise
