from pydantic import BaseModel, Field
from anthropic import Anthropic
from anthropic.types import Usage
from langfuse import Langfuse
from dotenv import load_dotenv
from models.types import PainScore, ServiceScore
from prompts.templates import service_prompt, pain_eval_prompt, service_eval_prompt
import uuid
import instructor
import polars as pl

load_dotenv()
langfuse = Langfuse()


class PainService(BaseModel):
    pain_point: str
    service_offering: str


class EvalSet(BaseModel):
    original_text: str
    output: PainService


class Relevance(BaseModel):
    pain_relevance_score: int = Field(
        ...,
        description="Score between 0 and 5 of the relevance of pain point and service offering to the original job posting.",
    )
    service_offering_relevance_score: int = Field(
        ...,
        description="Score between 0 and 5 of the relevance of the service offering to the original job posting and the pain point.",
    )


client = instructor.from_anthropic(Anthropic())
eval_client = instructor.from_anthropic(Anthropic())

good_dataset = langfuse.create_dataset("good_data")
bad_dataset = langfuse.create_dataset("bad_data")


def run_generation(
    compiled_prompt,
    input_vars,
    trace_name,
    response_model: BaseModel,
    generation_step_name,
    trace_id,
    model="claude-3-haiku-20240307",
):
    trace = langfuse.trace(name=trace_name, id=trace_id)
    generation = trace.generation(name=generation_step_name, model=model)

    try:
        step, raw = client.messages.create_with_completion(
            model=model,
            messages=[{"role": "user", "content": compiled_prompt}],
            response_model=response_model,
            max_tokens=4000,
        )
        usage: Usage = raw.usage
        generation.end(
            input={"prompt": compiled_prompt, "vars": input_vars},
            output=step,
            level="DEFAULT",
            usage={"input": usage.input_tokens, "output": usage.output_tokens},
        )
        trace.update(input=compiled_prompt, output=step)
        langfuse.flush()
        return step, generation
    except Exception as e:
        print(e)
        generation.end(input=compiled_prompt, output=str(e), level="ERROR")
        trace.update(input=compiled_prompt, output=str(e))
        langfuse.flush()


def run_end_eval(
    compiled_prompt,
    input_vars,
    response_model: BaseModel,
    generation_step_name,
    trace_id,
    model="claude-3-haiku-20240307",
):
    trace = langfuse.trace(id=trace_id)
    generation = trace.generation(name=generation_step_name, model=model)
    try:
        step, raw = client.messages.create_with_completion(
            model=model,
            messages=[{"role": "user", "content": compiled_prompt}],
            response_model=response_model,
            max_tokens=4000,
        )
        usage: Usage = raw.usage
        generation.end(
            input={"prompt": compiled_prompt, "vars": input_vars},
            output=step,
            level="DEFAULT",
            usage={"input": usage.input_tokens, "output": usage.output_tokens},
        )
        trace.update(input=compiled_prompt, output=step)
        langfuse.flush()
        return step.score
    except Exception as e:
        print(e)
        generation.end(input=input_vars, output=str(e), level="ERROR")
        trace.update(input=compiled_prompt, output=str(e))
        langfuse.flush()


upwork_data = pl.read_csv("./upwork.csv").select(pl.col("description")).to_series()
trace_id = str(uuid.uuid4())


def run_app():
    for i, d in enumerate(upwork_data[:3]):
        prompt = service_prompt.format(job_description=d)
        job_description = {"job_description": d}
        try:
            upwork_value, generation = run_generation(
                prompt,
                job_description,
                "upwork_extract_pain_service",
                PainService,
                "pain_service",
                trace_id,
            )
            # Evaluate
            paint_finish = pain_eval_prompt.format(
                job_description=d, pain=upwork_value.pain_point
            )
            service_finish = service_eval_prompt.format(
                job_description=d, offering=upwork_value.service_offering
            )
            print("eval 1")
            generation.score(
                name="pain_score",
                value=run_end_eval(
                    compiled_prompt=paint_finish,
                    input_vars={
                        "job_description": d,
                        "pain_point": upwork_value.pain_point,
                    },
                    response_model=PainScore,
                    generation_step_name="pain_eval",
                    trace_id=trace_id,
                ),
            )
            print("eval 2")
            generation.score(
                name="service_score",
                value=run_end_eval(
                    compiled_prompt=service_finish,
                    input_vars={
                        "job_description": d,
                        "service_offering": upwork_value.service_offering,
                    },
                    response_model=ServiceScore,
                    generation_step_name="service_eval",
                    trace_id=trace_id,
                ),
            )
            langfuse.create_dataset_item(
                "good_data",
                input={"prompt": prompt, "job_description": job_description},
                expected_output={"output": upwork_value},
                source_trace_id=trace_id,
                source_observation_id=generation.id,
            )
        except Exception as e:
            langfuse.create_dataset_item(
                "bad_data",
                input={"prompt": prompt, "job_description": job_description},
                expected_output={"error": e},
            )


run_app()
