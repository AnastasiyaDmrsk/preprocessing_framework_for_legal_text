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

if __name__ == "__main__":
    role_task_mapping="""
    <?xml version="1.0" encoding="UTF-8"?>
<tasks>

  <task id="t1">
    <label>provide SMEs with priority access to the AI regulatory sandboxes</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions>
      <condition>if SMEs fulfil the eligibility conditions and selection criteria</condition>
    </conditions>
    <exceptions>
      <exception ref="t1" description="Priority access shall not preclude other SMEs, including start-ups, other than those referred to in paragraph 1 from access to the AI regulatory sandbox, provided that they also fulfil the eligibility conditions and selection criteria"/>
    </exceptions>
    <source-ref article="62" paragraph="1"/>
  </task>

  <task id="t2">
    <label>organise specific activities</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="1"/>
  </task>

  <task id="t3">
    <label>utilise existing dedicated channels</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="1"/>
  </task>

  <task id="t4">
    <label>establish new channels</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="1"/>
  </task>

  <task id="t5">
    <label>facilitate the participation of SMEs in the development process</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="1"/>
  </task>

  <task id="t6">
    <label>share interests and needs</label>
    <performers>
      <performer><unit>SME</unit><role>SME</role></performer>
    </performers>
    <deontic type="permission" modality="may"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="2"/>
  </task>

  <task id="t7">
    <label>take into account the specific interests and needs</label>
    <performers>
      <performer><unit>Member State</unit><role>Member State</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="2"/>
  </task>

  <task id="t8">
    <label>request standardised templates</label>
    <performers>
      <performer><unit>Board</unit><role>Board</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t9">
    <label>provide standardised templates</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions>
      <condition>as specified by the Board in its request</condition>
    </conditions>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t10">
    <label>develop a single information platform</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t11">
    <label>maintain a single information platform</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t12">
    <label>organise appropriate communication campaigns</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t13">
    <label>evaluate the convergence of best practices</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

  <task id="t14">
    <label>promote the convergence of best practices</label>
    <performers>
      <performer><unit>AI Office</unit><role>AI Office</role></performer>
    </performers>
    <deontic type="obligation" modality="shall"/>
    <conditions/>
    <exceptions/>
    <source-ref article="62" paragraph="3"/>
  </task>

</tasks>
    """
    preprocessed_text = """
    Article 62

Measures for providers and deployers, in particular SMEs, including start-ups

1.   Member States shall undertake the following actions:
(a)
provide SMEs, including start-ups, having a registered office or a branch in the Union, with priority access to the AI regulatory sandboxes, to the extent that they fulfil the eligibility conditions and selection criteria; the priority access shall not preclude other SMEs, including start-ups, other than those referred to in this paragraph from access to the AI regulatory sandbox, provided that they also fulfil the eligibility conditions and selection criteria;
(b)
organise specific awareness raising and training activities on the application of this Regulation tailored to the needs of SMEs including start-ups, deployers and, as appropriate, local public authorities;
(c)
utilise existing dedicated channels and where appropriate, establish new ones for communication with SMEs including start-ups, deployers, other innovators and, as appropriate, local public authorities to provide advice and respond to queries about the implementation of this Regulation, including as regards participation in AI regulatory sandboxes;
(d)
facilitate the participation of SMEs and other relevant stakeholders in the standardisation development process.
2.   The specific interests and needs of the SME providers, including start-ups, shall be taken into account when setting the fees for conformity assessment under Article 43, reducing those fees proportionately to their size, market size and other relevant indicators.
3.   The AI Office shall undertake the following actions:
(a)
provide standardised templates for areas covered by this Regulation, as specified by the Board in its request;
(b)
develop and maintain a single information platform providing easy to use information in relation to this Regulation for all operators across the Union;
(c)
organise appropriate communication campaigns to raise awareness about the obligations arising from this Regulation;
(d)
evaluate and promote the convergence of best practices in public procurement procedures in relation to AI systems.
    """
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    description = build_process_description(
        preprocessed_text=preprocessed_text,
        role_task_mapping=role_task_mapping,
        api_key="",
        output_dir=output_dir,
    )
    print(description)