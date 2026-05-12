import warnings
from typing import List

from src.pipeline.role_task_mapping.const import (
    TASK_NLP_CANDIDATES_PROMPT,
    DEFAULT_TASK_EXTRACTION_MAX_TOKENS,
)
from src.pipeline.role_task_mapping.llm_task_extractor import LLMTaskExtractor
from src.pipeline.role_task_mapping.nlp_task_extractor import NLPTaskCandidateExtractor
from src.pipeline.role_task_mapping.models import Task, TaskCandidate
from src.pipeline.organigram.const import DEFAULT_NLP_MIN_CANDIDATES, DEFAULT_VALIDATOR_MODEL, DEFAULT_SPACY_MODEL, DEFAULT_MODEL


def _serialize_candidates_for_prompt(
        candidates: List[TaskCandidate]
) -> str:
    blocks: List[str] = []
    for c in candidates:
        conditions_str = " ;; ".join(c.conditions) or "(none)"
        blocks.append(
            f"---CANDIDATE---\n"
            f"label: {c.label}\n"
            f"deontic_type: {c.deontic_type}\n"
            f"deontic_modality: {c.deontic_modality}\n"
            f"conditions: {conditions_str}\n"
            f"article: {c.source_article}\n"
            f"paragraph: {c.source_paragraph}\n"
            f"---END---"
        )
    return "\n\n".join(blocks)


class HybridTaskExtractor(LLMTaskExtractor):

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        validator_model: str = DEFAULT_VALIDATOR_MODEL,
        use_validator: bool = True,
        nlp_min_candidates: int = DEFAULT_NLP_MIN_CANDIDATES,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            validator_model=validator_model,
            use_validator=use_validator,
        )
        self.nlp_extractor = NLPTaskCandidateExtractor(spacy_model=spacy_model)
        self.nlp_min_candidates = nlp_min_candidates

    def _extract_tasks(self, text: str, organigram_xml: str) -> List[Task]:
        candidates = self.nlp_extractor.extract_candidates(text)

        if len(candidates) < self.nlp_min_candidates:
            warnings.warn(
                f"Only {len(candidates)} NLP candidates found "
                f"(min={self.nlp_min_candidates}). Falling back to pure LLM.",
                RuntimeWarning,
            )
            return super()._extract_tasks(text, organigram_xml)

        return self._llm_validate_candidates(candidates, text, organigram_xml)

    def _llm_validate_candidates(
        self,
        candidates: List[TaskCandidate],
        text: str,
        organigram_xml: str,
    ) -> List[Task]:
        subjects_str = self._serialize_organigram_subjects(organigram_xml)
        candidates_str = _serialize_candidates_for_prompt(candidates)

        prompt = (
            TASK_NLP_CANDIDATES_PROMPT
            + "\n\nORGANIGRAM SUBJECTS:\n"
            + subjects_str
            + "\n\nNLP PRE-EXTRACTED TASK CANDIDATES:\n"
            + candidates_str
            + "\n\nINPUT TEXT:\n"
            + text.strip()
        )
        response = self._call(prompt, DEFAULT_TASK_EXTRACTION_MAX_TOKENS)
        return self._parse_task_blocks(response)

