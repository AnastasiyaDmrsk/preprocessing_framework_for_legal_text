from pathlib import Path
from typing import Tuple, List

from .preprocess import RegulatoryTextPreprocessor


def preprocess_legal_text(reg_text: str, path: Path) -> Tuple[str, List[str]]:
    """
    Preprocess regulatory text and write outputs to *path*.

    Outputs
    -------
    preprocess.txt  — one obligation clause per line, with gateway markers
    references.csv  — Location,Reference rows for all detected cross-references

    Returns
    -------
    (preprocessed_text, references_list)
    """
    reg_preprocessor       = RegulatoryTextPreprocessor()
    preprocessed_text, references = reg_preprocessor.preprocess(reg_text)

    path.mkdir(parents=True, exist_ok=True)
    (path / "preprocess.txt").write_text(preprocessed_text, encoding="utf-8")
    with open(path / "references.csv", "w", encoding="utf-8") as f:
        f.write("Location,Reference\n")
        f.writelines(ref + "\n" for ref in references)

    return preprocessed_text, references

if __name__ == "__main__":
    input_text = """
    Article 23

Reporting obligations

1.   Each Member State shall ensure that essential and important entities notify, without undue delay, its CSIRT or, where applicable, its competent authority in accordance with paragraph 4 of any incident that has a significant impact on the provision of their services as referred to in paragraph 3 (significant incident). Where appropriate, entities concerned shall notify, without undue delay, the recipients of their services of significant incidents that are likely to adversely affect the provision of those services. Each Member State shall ensure that those entities report, inter alia, any information enabling the CSIRT or, where applicable, the competent authority to determine any cross-border impact of the incident. The mere act of notification shall not subject the notifying entity to increased liability.
Where the entities concerned notify the competent authority of a significant incident under the first subparagraph, the Member State shall ensure that that competent authority forwards the notification to the CSIRT upon receipt.
In the case of a cross-border or cross-sectoral significant incident, Member States shall ensure that their single points of contact are provided in due time with relevant information notified in accordance with paragraph 4.
2.   Where applicable, Member States shall ensure that essential and important entities communicate, without undue delay, to the recipients of their services that are potentially affected by a significant cyber threat any measures or remedies that those recipients are able to take in response to that threat. Where appropriate, the entities shall also inform those recipients of the significant cyber threat itself.
3.   An incident shall be considered to be significant if:
(a)
it has caused or is capable of causing severe operational disruption of the services or financial loss for the entity concerned;
(b)
it has affected or is capable of affecting other natural or legal persons by causing considerable material or non-material damage.
4.   Member States shall ensure that, for the purpose of notification under paragraph 1, the entities concerned submit to the CSIRT or, where applicable, the competent authority:
(a)
without undue delay and in any event within 24 hours of becoming aware of the significant incident, an early warning, which, where applicable, shall indicate whether the significant incident is suspected of being caused by unlawful or malicious acts or could have a cross-border impact;
(b)
without undue delay and in any event within 72 hours of becoming aware of the significant incident, an incident notification, which, where applicable, shall update the information referred to in point (a) and indicate an initial assessment of the significant incident, including its severity and impact, as well as, where available, the indicators of compromise;
(c)
upon the request of a CSIRT or, where applicable, the competent authority, an intermediate report on relevant status updates;
(d)
a final report not later than one month after the submission of the incident notification under point (b), including the following:
(i)
a detailed description of the incident, including its severity and impact;
(ii)
the type of threat or root cause that is likely to have triggered the incident;
(iii)
applied and ongoing mitigation measures;
(iv)
where applicable, the cross-border impact of the incident;
(e)
in the event of an ongoing incident at the time of the submission of the final report referred to in point (d), Member States shall ensure that entities concerned provide a progress report at that time and a final report within one month of their handling of the incident.
By way of derogation from the first subparagraph, point (b), a trust service provider shall, with regard to significant incidents that have an impact on the provision of its trust services, notify the CSIRT or, where applicable, the competent authority, without undue delay and in any event within 24 hours of becoming aware of the significant incident.
5.   The CSIRT or the competent authority shall provide, without undue delay and where possible within 24 hours of receiving the early warning referred to in paragraph 4, point (a), a response to the notifying entity, including initial feedback on the significant incident and, upon request of the entity, guidance or operational advice on the implementation of possible mitigation measures. Where the CSIRT is not the initial recipient of the notification referred to in paragraph 1, the guidance shall be provided by the competent authority in cooperation with the CSIRT. The CSIRT shall provide additional technical support if the entity concerned so requests. Where the significant incident is suspected to be of criminal nature, the CSIRT or the competent authority shall also provide guidance on reporting the significant incident to law enforcement authorities.
    """
    preprocessor = RegulatoryTextPreprocessor()
    preprocessed, refs = preprocessor.preprocess(input_text)
    print(preprocessed)
    print(refs)
