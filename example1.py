import textwrap

import langextract as lx

from config import config

prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"},
            ),
        ],
    )
]

input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id=config.MODEL_NAME,
    api_key=config.API_KEY.get_secret_value(),
)

lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl")

html_content = lx.visualize("test_output/extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)
