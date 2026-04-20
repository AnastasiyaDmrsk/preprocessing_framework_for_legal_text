from pathlib import Path

from src.pipeline.organigram.const import DEFAULT_MODEL, DEFAULT_SPACY_MODEL, DEFAULT_VALIDATOR_MODEL
from src.pipeline.organigram.llm_extractor import LLMOrganizationalExtractor
from src.pipeline.organigram.organization import HybridOrganizationalExtractor


def build_organigram_xml(
        preprocessed_text: str,
        api_key: str,
        output_dir: Path,
        model: str = DEFAULT_MODEL,
        use_hybrid: bool = True,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        use_gliner: bool = False,
        use_validator: bool = True,
        validator_model: str = DEFAULT_VALIDATOR_MODEL,
) -> str:
    extractor = (
        HybridOrganizationalExtractor(
            api_key=api_key,
            model=model,
            spacy_model=spacy_model,
            use_gliner=use_gliner,
            use_validator=use_validator,
            validator_model=validator_model,
        )
        if use_hybrid
        else LLMOrganizationalExtractor(
            api_key=api_key,
            model=model,
            use_validator=use_validator,
            validator_model=validator_model,
        )
    )
    return extractor.extract_and_save_organigram(
        preprocessed_text=preprocessed_text,
        output_path=output_dir / "organigram.xml",
    )
