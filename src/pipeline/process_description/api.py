from pathlib import Path
from typing import Union

from src.pipeline.process_description.const import PROCESS_DESCRIPTION_FILENAME
from src.pipeline.process_description.nlp_extractor import ProcessDescriptionNLPExtractor
from src.pipeline.process_description.llm_generator import LLMProcessDescriptionGenerator
from src.pipeline.organigram.const import DEFAULT_MODEL, DEFAULT_VALIDATOR_MODEL


def build_process_description(
    preprocessed_text: str,
    role_task_mapping: Union[str, Path],
    api_key: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    use_validator: bool = True,
    validator_model: str = DEFAULT_VALIDATOR_MODEL,
) -> str:
    xml_content = (
        role_task_mapping.read_text(encoding="utf-8")
        if isinstance(role_task_mapping, Path)
        else role_task_mapping
    )

    extractor = ProcessDescriptionNLPExtractor()
    tasks = extractor.extract_tasks(xml_content)

    generator = LLMProcessDescriptionGenerator(
        api_key=api_key,
        model=model,
        validator_model=validator_model,
        use_validator=use_validator,
    )

    return generator.generate_and_save(
        preprocessed_text=preprocessed_text,
        tasks=tasks,
        output_path=output_dir / PROCESS_DESCRIPTION_FILENAME,
    )
