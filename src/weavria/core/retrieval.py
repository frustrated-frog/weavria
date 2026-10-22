"""Versioned, content-free evidence of scoped historical discovery."""

from typing import Literal, Self

from pydantic import Field, model_validator

from ._schema import ContractModel, NonBlankString


class HistorySourceAddress(ContractModel):
    """An exact range in accepted conversation text, never a new execution fact."""

    session_id: NonBlankString
    run_id: NonBlankString
    ordinal: int = Field(ge=1)
    accepted_sequence: int = Field(ge=1)
    field: Literal["user", "assistant", "update"]
    update_id: NonBlankString | None = None
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end <= self.start or (self.field == "update") != (self.update_id is not None):
            raise ValueError(
                "historical address requires a nonempty range and exact field identity"
            )
        return self


class HistoryMatch(ContractModel):
    address: HistorySourceAddress
    demand_ids: tuple[NonBlankString, ...]
    score: float = Field(ge=0, allow_inf_nan=False)


class RetrievedHistoryCandidate(ContractModel):
    run_id: NonBlankString
    ordinal: int = Field(ge=1)
    view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin: Literal["recent", "lexical", "anchor"]


class HistoryRetrievalReport(ContractModel):
    """Search coverage and delivery are separate; the request records actual delivery."""

    policy_version: Literal["history-bm25/1"] = "history-bm25/1"
    session_id: NonBlankString
    task_revision: int = Field(ge=1)
    before_ordinal: int = Field(ge=1)
    through_sequence: int = Field(ge=0)
    query_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_generation: int = Field(ge=1)
    indexed_through_ordinal: int = Field(ge=0)
    indexed_through_sequence: int = Field(ge=0)
    index_complete: bool
    search_complete: bool
    stop_reason: Literal[
        "complete", "index_budget", "query_budget", "query_terms", "time_budget", "candidate_budget"
    ]
    indexed_turns: int = Field(ge=0)
    postings_examined: int = Field(ge=0)
    query_terms: int = Field(ge=0)
    candidates: tuple[RetrievedHistoryCandidate, ...] = Field(max_length=256)
    matches: tuple[HistoryMatch, ...] = Field(max_length=1024)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        ids = {candidate.run_id for candidate in self.candidates}
        ordinals = {candidate.run_id: candidate.ordinal for candidate in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("retrieval candidates must be unique")
        if self.stop_reason == "complete" and not (self.index_complete and self.search_complete):
            raise ValueError("complete retrieval requires complete index and search")
        if any(c.ordinal >= self.before_ordinal for c in self.candidates):
            raise ValueError("retrieval candidate is outside the historical boundary")
        for match in self.matches:
            address = match.address
            if (
                address.session_id != self.session_id
                or address.ordinal >= self.before_ordinal
                or address.accepted_sequence > self.through_sequence
                or address.run_id not in ids
                or address.ordinal != ordinals.get(address.run_id)
            ):
                raise ValueError("retrieval match is outside its accepted source scope")
        return self
