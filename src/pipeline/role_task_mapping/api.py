from pathlib import Path

from src.pipeline.role_task_mapping.llm_task_extractor import LLMTaskExtractor
from src.pipeline.role_task_mapping.role_task_mapper import HybridTaskExtractor
from src.pipeline.organigram.const import DEFAULT_MODEL, DEFAULT_SPACY_MODEL, DEFAULT_VALIDATOR_MODEL


def build_role_task_mapping(
    preprocessed_text: str,
    organigram: str,
    api_key: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    use_hybrid: bool = True,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    use_validator: bool = True,
    validator_model: str = DEFAULT_VALIDATOR_MODEL,
) -> str:
    extractor = (
        HybridTaskExtractor(
            api_key=api_key,
            model=model,
            spacy_model=spacy_model,
            use_validator=use_validator,
            validator_model=validator_model,
        )
        if use_hybrid
        else LLMTaskExtractor(
            api_key=api_key,
            model=model,
            use_validator=use_validator,
            validator_model=validator_model,
        )
    )
    return extractor.extract_and_save_tasks(
        text=preprocessed_text,
        organigram_xml=organigram,
        output_path=output_dir / "role_task_mapping.xml",
    )
