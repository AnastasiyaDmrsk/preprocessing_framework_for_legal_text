PROP_NS = "http://cpee.org/ns/properties/2.0"
DESC_NS = "http://cpee.org/ns/description/1.0"
ANNO_NS = "http://cpee.org/ns/annotation/1.0"

TESTSET_FILENAME = "testset.xml"
WORKLIST_FILENAME = "worklist.xml"

DEFAULT_CPEE_BASE = "https://cpee.org/flow/engine/"
DEFAULT_FORM_URL = "https://cpee.org/~demo/form/form-f.html"

DEFAULT_GENERATOR_MAX_TOKENS = 8192
DEFAULT_GENERATOR_TEMPERATURE = 0.2
DEFAULT_VALIDATOR_MAX_TOKENS = 8192
DEFAULT_VALIDATOR_TEMPERATURE = 0.0

TESTSET_DESCRIPTION_PROMPT = """You are an expert in CPEE process modelling. Convert a \
TEXTUAL process description plus a role/task mapping into a single CPEE description block.

PROCESS DESCRIPTION (free text):
[PROCESS_DESCRIPTION]

ROLE/TASK MAPPING (authoritative source of roles):
[ROLE_TASK_MAPPING]

OUTPUT
Return ONLY the description element, nothing else — no prose, no comments, no code fences:
<description xmlns="http://cpee.org/ns/description/1.0"> ... </description>

CONTROL-FLOW VOCABULARY (use exactly these elements)
- Sequence: list elements in execution order as siblings.
- Atomic task: a <call> (see CALL FORMAT below). One <call> per concrete activity.
- AND gateway (things happen in parallel / "in parallel" / "at the same time"):
    <parallel wait="-1" cancel="last">
      <parallel_branch> ...one branch... </parallel_branch>
      <parallel_branch> ...another branch... </parallel_branch>
    </parallel>
- XOR gateway (a decision / "if ... else" / "where appropriate" / mutually exclusive paths):
    <choose mode="exclusive">
      <alternative condition="SHORT CONDITION FROM THE TEXT"> ...path... </alternative>
      <alternative condition="THE OTHER CONDITION"> ...path... </alternative>
      <otherwise> ...optional default path... </otherwise>
    </choose>
  Use <otherwise> only for an explicit "otherwise/else" with no stated condition.
- Loop (repeat while/until): <loop mode="pre_test" condition="CONDITION"> ...body... </loop>

CALL FORMAT — every task is exactly:
    <call id="aN" endpoint="worklist">
      <parameters>
        <label>SHORT TASK LABEL</label>
        <arguments><role>ROLE</role></arguments>
      </parameters>
    </call>

RULES
- Derive the structure faithfully from the text: model every parallel block as <parallel>,
  every decision as <choose>, and preserve the order described. Branches that rejoin ("after
  all these are merged", "following either path") simply close the gateway and continue as
  siblings. Start/end events are NOT tasks — do not emit calls for them.
- Mint unique ids a1, a2, a3, ... in document order.
- <label> must be a concise activity name (imperative, e.g. "Provide advice"), matching the
  wording in the text/mapping — not a whole sentence.
- ROLE: pick the role of the mapping entry whose label is most semantically similar to the
  task. Use ONLY roles that appear in the mapping; never invent a role.
- Do not add tasks that are not in the text, and do not drop any described activity.

EXAMPLE (shows the gateway mapping — not related to the input above)
Text: "A does X. Then, if risk is high B does Y, otherwise B does Z. Finally A and B work in
parallel on P and Q respectively."
Output:
<description xmlns="http://cpee.org/ns/description/1.0">
  <call id="a1" endpoint="worklist"><parameters><label>X</label><arguments><role>A</role></arguments></parameters></call>
  <choose mode="exclusive">
    <alternative condition="Risk is high"><call id="a2" endpoint="worklist"><parameters><label>Y</label><arguments><role>B</role></arguments></parameters></call></alternative>
    <otherwise><call id="a3" endpoint="worklist"><parameters><label>Z</label><arguments><role>B</role></arguments></parameters></call></otherwise>
  </choose>
  <parallel wait="-1" cancel="last">
    <parallel_branch><call id="a4" endpoint="worklist"><parameters><label>P</label><arguments><role>A</role></arguments></parameters></call></parallel_branch>
    <parallel_branch><call id="a5" endpoint="worklist"><parameters><label>Q</label><arguments><role>B</role></arguments></parameters></call></parallel_branch>
  </parallel>
</description>

Return the valid XML now."""

TESTSET_DESCRIPTION_VALIDATION_PROMPT = """You are a strict auditor of CPEE process modelling.
Validate and, if needed, correct a CPEE description \
block generated from a textual process description. Check the validity of the XML structure.

PROCESS DESCRIPTION (free text — ground truth for structure):
[PROCESS_DESCRIPTION]

ROLE/TASK MAPPING (ground truth for roles):
[ROLE_TASK_MAPPING]

PROPOSED BLOCK:
[PROPOSED_DESCRIPTION]

CHECKS TO PERFORM (reason through each explicitly before outputting):
1. Every activity in the text appears as exactly one <call>, with nothing invented or
dropped, and start/end events are not modelled as calls; 
2. Every "in parallel" block is a <parallel> with one <parallel_branch> per concurrent strand, and every decision is a <choose>
with <alternative condition="..."> paths (and <otherwise> only for an explicit default);
3. Ordering and branch rejoining match the text; 
4. Each call has a unique id (a1, a2, ...), endpoint="worklist", a concise <label>, and a <role> drawn only from the mapping that best fits
the task; 
5. The XML is well formed and valid. 

Fix any issues identified, then return the final block as:
<FINAL_DESCRIPTION>
<description xmlns="http://cpee.org/ns/description/1.0"> ... </description>
</FINAL_DESCRIPTION>

EXAMPLE (shows the gateway mapping — not related to the input above)
Text: "A does X. Then, if risk is high B does Y, otherwise B does Z. Finally A and B work in
parallel on P and Q respectively."
Output:
<description xmlns="http://cpee.org/ns/description/1.0">
  <call id="a1" endpoint="worklist"><parameters><label>X</label><arguments><role>A</role></arguments></parameters></call>
  <choose mode="exclusive">
    <alternative condition="Risk is high"><call id="a2" endpoint="worklist"><parameters><label>Y</label><arguments><role>B</role></arguments></parameters></call></alternative>
    <otherwise><call id="a3" endpoint="worklist"><parameters><label>Z</label><arguments><role>B</role></arguments></parameters></call></otherwise>
  </choose>
  <parallel wait="-1" cancel="last">
    <parallel_branch><call id="a4" endpoint="worklist"><parameters><label>P</label><arguments><role>A</role></arguments></parameters></call></parallel_branch>
    <parallel_branch><call id="a5" endpoint="worklist"><parameters><label>Q</label><arguments><role>B</role></arguments></parameters></call></parallel_branch>
  </parallel>
</description>
"""

_TESTSET_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<testset xmlns="{prop}">
  <executionhandler>ruby</executionhandler>
  <dataelements/>
  <endpoints>
    <user>https-post://cpee.org/services/timeout-user.php</user>
    <worklist>{worklist_ep}</worklist>
    <auto>https-post://cpee.org/services/timeout-auto.php</auto>
    <subprocess>https-post://cpee.org/flow/start/url/</subprocess>
    <timeout>https-post://cpee.org/services/timeout.php</timeout>
    <send>https-post://cpee.org/ing/correlators/message/send/</send>
    <receive>https-get://cpee.org/ing/correlators/message/receive/</receive>
  </endpoints>
  <attributes>
    <info>Worklist</info>
    <modeltype>CPEE</modeltype>
    <theme>extended</theme>
    <organisation1>{orgmodel}</organisation1>
    <creator>AutoBPMN</creator>
    <author>AutoBPMN</author>
    <design_stage>development</design_stage>
    <design_dir>Templates.dir</design_dir>
  </attributes>
  <description>
{inner}
  </description>
  <transformation>
    <description type="copy"/>
    <dataelements type="none"/>
    <endpoints type="none"/>
  </transformation>
</testset>
"""