from __future__ import annotations

from pathlib import Path
from typing import Union

from src.pipeline.AutoBPMN_testset.const import DEFAULT_FORM_URL, DEFAULT_CPEE_BASE, WORKLIST_FILENAME, TESTSET_FILENAME
from src.pipeline.AutoBPMN_testset.testset_builder import parse_role_mapping, cpee_endpoint, organigram_path, \
    worklist_path, serialize_role_mapping, assemble_testset, build_worklist_xml, enrich_structure, \
    extract_roles_from_block, RoleAssigner, LLMTestsetDescriptionGenerator, plain_url, _read


def build_testset(process_description: Union[str, Path], role_task_mapping: Union[str, Path], api_key: str,
                  output_dir: Path, *, base_url: str, job_id: str, model: str, use_validator: bool = True,
                  validator_model: str = None, form_url: str = DEFAULT_FORM_URL, cpee_base: str = DEFAULT_CPEE_BASE,
                  process_name: str = "Worklist", trust_llm_structure: bool = True, ) -> str:
    """
    Write testset.xml and worklist.xml for one job; return the testset XML string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    proc_text = _read(process_description)
    mapping = parse_role_mapping(role_task_mapping)

    orgmodel_url = plain_url(base_url, organigram_path(job_id))
    worklist_ep = cpee_endpoint(base_url, worklist_path(job_id), "post")

    generator = LLMTestsetDescriptionGenerator(api_key=api_key, model=model, validator_model=validator_model,
                                               use_validator=use_validator, )
    llm_block = generator.generate(proc_text, serialize_role_mapping(mapping))
    by_id, by_label = extract_roles_from_block(llm_block)
    assigner = RoleAssigner(mapping, by_id, by_label)

    structure = llm_block if trust_llm_structure else proc_text
    inner_xml, calls = enrich_structure(structure, role_for=assigner.resolve, orgmodel_url=orgmodel_url,
                                        form_url=form_url, )

    worklist_xml = build_worklist_xml(calls, orgmodel_url=orgmodel_url, cpee_base=cpee_base, process_name=process_name,
                                      job_id=job_id, )
    (output_dir / WORKLIST_FILENAME).write_text(worklist_xml, encoding="utf-8")

    testset = assemble_testset(inner_xml, worklist_endpoint=worklist_ep, orgmodel_url=orgmodel_url, )
    (output_dir / TESTSET_FILENAME).write_text(testset, encoding="utf-8")
    return testset
