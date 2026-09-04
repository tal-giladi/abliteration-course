# Reference solution for option (a): a length-based coherence sanity check,
# added as a second scorer alongside KLDivergence.
#
# The file below is what you'd create at
# lab/vendor/heretic/src/heretic/scorers/response_length.py. It's included
# here as a string so this solution file stays importable and self-contained
# without requiring lab/vendor/heretic to exist on the machine grading it.

RESPONSE_LENGTH_SCORER_SOURCE = '''\
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:50]",
            column="text",
        ),
        description="Prompts to measure response length on.",
    )
    target_length: int = Field(
        default=60,
        description="Expected response length in tokens, for comparison.",
    )


class ResponseLength(Scorer):
    """Flags responses that collapsed to near-nothing or ballooned far past a
    normal length -- a cheap, fast coherence sanity check independent of KL
    divergence."""

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Response length ratio"

    def init(self, ctx: Context) -> None:
        self.prompts = ctx.load_prompts(self.settings.prompts)

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        lengths = [len(response.split()) for response in responses]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        ratio = avg_length / self.settings.target_length
        return Score(
            value=abs(1.0 - ratio),
            rich_display=f"[bold]{ratio:.2f}[/]x target",
            md_display=f"{ratio:.2f}x target",
        )
'''


def describe_change() -> dict:
    return {
        "option": "scorer",
        "module": "heretic.scorers.response_length",
        "class_name": "ResponseLength",
    }
