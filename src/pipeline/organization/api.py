from pathlib import Path

from .const import DEFAULT_MODEL, DEFAULT_SPACY_MODEL
from .llm_extractor import LLMOrganizationalExtractor
from .organization import HybridOrganizationalExtractor


def build_organigram_xml(
    preprocessed_text: str,
    api_key: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    use_hybrid: bool = True,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    use_gliner: bool = False,
) -> str:
    """
    Build organigram.xml from preprocessed text.
    :param preprocessed_text: preprocessed text from the previous step
    :param api_key: api key to use for LLM
    :param output_dir: directory to save organigram.xml
    :param model: LLM to use
    :param use_hybrid: use hybrid extractor or only LLM extractor
    :param spacy_model: spacy model to use for NLP candidate extraction (if hybrid)
    :param use_gliner: whether to add GLiNER zero-shot layer to the NLP pipeline (slower, higher recall)

    Args:
        use_hybrid: True  → NLP candidate extraction + LLM validation
                    False → pure LLM extraction
        use_gliner: add GLiNER zero-shot layer to the NLP pipeline (slower, higher recall)
    """
    extractor = (
        HybridOrganizationalExtractor(
            api_key=api_key,
            model=model,
            spacy_model=spacy_model,
            use_gliner=use_gliner,
        )
        if use_hybrid
        else LLMOrganizationalExtractor(api_key=api_key, model=model)
    )
    return extractor.extract_and_save_organigram(
        preprocessed_text=preprocessed_text,
        output_path=output_dir / "organigram.xml",
    )
