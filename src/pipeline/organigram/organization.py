from typing import List, Tuple

from src.pipeline.organigram.const import (
    DEFAULT_MODEL,
    DEFAULT_SPACY_MODEL,
    DEFAULT_GLINER_MODEL,
    DEFAULT_GLINER_THRESHOLD,
    DEFAULT_NLP_MIN_CANDIDATES,
    DEFAULT_VALIDATOR_MODEL,
)
from src.pipeline.organigram.llm_extractor import LLMOrganizationalExtractor
from src.pipeline.organigram.nlp_extractor import NLPActorCandidateExtractor
from src.pipeline.organigram.utils import _build_pre_extracted_actors_hierarchy_prompt


class HybridOrganizationalExtractor(LLMOrganizationalExtractor):

    def __init__(
            self,
            api_key: str,
            model: str = DEFAULT_MODEL,
            spacy_model: str = DEFAULT_SPACY_MODEL,
            use_gliner: bool = False,
            gliner_model: str = DEFAULT_GLINER_MODEL,
            gliner_threshold: float = DEFAULT_GLINER_THRESHOLD,
            nlp_min_candidates: int = DEFAULT_NLP_MIN_CANDIDATES,
            use_validator: bool = True,
            validator_model: str = DEFAULT_VALIDATOR_MODEL,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            validator_model=validator_model,
            use_validator=use_validator,
        )
        self.nlp_extractor = NLPActorCandidateExtractor(
            spacy_model=spacy_model,
            use_gliner=use_gliner,
            gliner_model=gliner_model,
            gliner_threshold=gliner_threshold,
        )
        self.nlp_min_candidates = nlp_min_candidates

    def _extract_actors(self, text: str) -> List[Tuple[str, str]]:
        nlp_candidates = self.nlp_extractor.extract_candidates(text)

        if len(nlp_candidates) < self.nlp_min_candidates:
            return super()._extract_actors(text)

        return self._llm_validate_candidates(nlp_candidates, text)

    def _llm_validate_candidates(
            self,
            nlp_candidates: List[Tuple[str, str]],
            text: str,
    ) -> List[Tuple[str, str]]:
        candidate_lines = "\n".join(
            f"- {name} | {t}" for name, t in nlp_candidates
        )
        prompt = _build_pre_extracted_actors_hierarchy_prompt(candidate_lines, text)
        response_text = self.generate_content(prompt, 4096)
        return self._parse_actor_response(response_text)
