import re
import warnings
from pathlib import Path
from typing import List

import google.genai as genai

from .const import (
    DEFAULT_GENERATOR_MAX_TOKENS,
    DEFAULT_GENERATOR_TEMPERATURE,
    DEFAULT_VALIDATOR_MAX_TOKENS,
    DEFAULT_VALIDATOR_TEMPERATURE,
    PROCESS_DESCRIPTION_PROMPT,
    PROCESS_DESCRIPTION_VALIDATION_PROMPT,
)
from .models import ExtractedTask
from .utils import serialize_tasks


class LLMProcessDescriptionGenerator:

    def __init__(
            self,
            api_key: str,
            model: str,
            validator_model: str = None,
            use_validator: bool = True,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.validator_model = validator_model or model
        self.use_validator = use_validator

    def generate_and_save(
            self,
            preprocessed_text: str,
            tasks: List[ExtractedTask],
            output_path: Path,
    ) -> str:
        serialized = serialize_tasks(tasks)
        description = self._generate(preprocessed_text, serialized)

        if self.use_validator:
            description = self._validate(description, preprocessed_text, serialized)

        output_path.write_text(description, encoding="utf-8")
        return description

    def _generate(self, preprocessed_text: str, serialized_tasks: str) -> str:
        prompt = (
            PROCESS_DESCRIPTION_PROMPT
            .replace("[ROLE TASK MAPPING]", serialized_tasks)
            .replace("[REGULATORY TEXT]", preprocessed_text.strip())
        )
        return self._call(prompt, DEFAULT_GENERATOR_MAX_TOKENS, DEFAULT_GENERATOR_TEMPERATURE)

    def _validate(
            self,
            description: str,
            preprocessed_text: str,
            serialized_tasks: str,
    ) -> str:
        prompt = (
            PROCESS_DESCRIPTION_VALIDATION_PROMPT
            .replace("[ROLE TASK MAPPING]", serialized_tasks)
            .replace("[REGULATORY TEXT]", preprocessed_text.strip())
            .replace("[PROPOSED_DESCRIPTION]", description.strip())
        )
        response = self._call(prompt, DEFAULT_VALIDATOR_MAX_TOKENS, DEFAULT_VALIDATOR_TEMPERATURE,
                              model=self.validator_model)

        match = re.search(
            r"<FINAL_DESCRIPTION>(.*?)</FINAL_DESCRIPTION>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        warnings.warn(
            "Process description validator missing <FINAL_DESCRIPTION> tags. Using generator output.",
            RuntimeWarning,
        )
        return description

    def _call(self, prompt: str, max_tokens: int, temperature: float, model: str = None) -> str:
        response = self.client.models.generate_content(
            model=model or self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return "".join(
            p.text for p in response.candidates[0].content.parts if p.text
        ).strip()
