from pathlib import Path

from .llm_task_extractor import LLMTaskExtractor
from .role_task_mapper import HybridTaskExtractor
from ..organigram.const import DEFAULT_MODEL, DEFAULT_SPACY_MODEL, DEFAULT_VALIDATOR_MODEL


def build_role_task_mapping(
    preprocessed_text: str,
    organigram: str,
    api_key: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    use_hybrid: bool = True,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    use_validator: bool = False, # TODO: change to True when validator is ready for tasks
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

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("MODEL")

    SAMPLE_ORGANIGRAM = """<organisation xmlns="http://cpee.org/ns/organisation/1.0">
  <units>
    <unit id="Applicant">
      <permissions/>
    </unit>
    <unit id="Member State">
      <permissions/>
    </unit>
    <unit id="Commission">
      <permissions/>
    </unit>
  </units>
  <roles>
    <role id="Applicant">
      <permissions/>
    </role>
    <role id="Driving Licence Holder">
      <parent>Applicant</parent>
      <permissions/>
    </role>
    <role id="Member State">
      <permissions/>
    </role>
    <role id="Commission">
      <permissions/>
    </role>
  </roles>
  <subjects>
    <subject id="applicant_name applicant_surname" uid="anas">
      <relation unit="Applicant" role="Applicant"/>
    </subject>
    <subject id="commission_name commission_surname" uid="cncs">
      <relation unit="Commission" role="Commission"/>
    </subject>
    <subject id="ms_name ms_surname" uid="msnmss">
      <relation unit="Member State" role="Member State"/>
    </subject>
    <subject id="dlh_name dlh_surname" uid="dlhndlhs">
      <relation unit="Applicant" role="Driving Licence Holder"/>
    </subject>
  </subjects>
</organisation>"""

    SAMPLE_TEXT = """Article 11: Compliance with the minimum standards of physical and mental fitness
1 Member States shall, ensure.
A medical examination shall, however, be required by Member States in respect of applications for driving licences of categories C, CE, C1, C1E, D, D1, DE or D1E, regardless.
2 applicants for the renewal shall undergo a medical examination covering the medical conditions set out in Annex III.
This shall apply to the renewal of driving licences in category AM only if so required by the Member State in question .
--PARALLEL GATEWAY--
3(a) Notwithstanding paragraphs 1 and 2, and to the extent not otherwise provided for by Annex III, such as in the case of the appropriate assessment of eyesight for applicants for driving licences IAW Annex III, point 3, Member States may, for categories AM, A, A1, A2, B, B1 and BE, instead of requiring a medical examination, apply one or both of the following alternative measures require the applicant or holder of the driving licence to fill in a self-assessment form covering the medical conditions set out in Annex III when applying for the issuance or renewal of a driving licence; or
3(b) Notwithstanding paragraphs 1 and 2, and to the extent not otherwise provided for by Annex III, such as in the case of the appropriate assessment of eyesight for applicants for driving licences IAW Annex III, point 3, Member States may, for categories AM, A, A1, A2, B, B1 and BE, instead of requiring a medical examination, apply one or both of the following alternative measures establish a national system of assessment of fitness to drive to ensure that significant changes in physical or mental fitness are reacted by Member States to in order to comply with the minimum standards of physical and mental fitness set out in Annex III, after the driving licence has been issued to the applicant following a medical examination or self-assessment
--END OF PARALLEL GATEWAY--
4 Member States may provide for appropriate measures or for knowingly providing information in the self-assessment form, or for failing to meet any requirement established IAW paragraph 3, point (b).
5 Member States may apply the alternative measure under paragraph 3, point (b), in such a way.
6 IF, on the basis of information acquired pursuant to the various alternative measures set out in paragraph 3, Member States becomes apparent that the applicant or holder of a driving licence is likely to have one or more of the medical conditions listed in Annex III, Member States shall ensure.
7 This Article shall not prevent Member States from taking measures.
IF Member States adopt guidelines for medical practitioners to help identify driving licence holders who no longer meet the minimum standards of physical and mental fitness to drive, Member States shall inform the Commission thereof.
The Commission shall make the guidelines available to the other Member States.
IF Member States develop public awareness campaigns to inform citizens about mental or physical health conditions that may impair fitness to drive, Member States shall inform the Commission thereof.
The Commission shall make the information available to the other Member States.
8 The standards set by Member States for the issuance or any subsequent renewal of driving licences may be stricter than those set out in Annex III."""

    out_dir = Path("test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_role_task_mapping(
        preprocessed_text=SAMPLE_TEXT,
        organigram=SAMPLE_ORGANIGRAM,
        api_key=api_key,
        output_dir=out_dir,
        model=model,
    )

    print("\n" + "=" * 80)
    print("ROLE TASK MAPPING XML")
    print("=" * 80)
    print(result)
    print(f"\nSaved to: {out_dir / 'role_task_mapping.xml'}")

