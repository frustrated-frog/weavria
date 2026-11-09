"""Model adapters for bounded proposal extraction and independent semantic review."""

from weavria.memory import (
    MemoryExtraction,
    MemoryFormationInput,
    MemoryReconciliation,
    MemoryReview,
)
from weavria.models import ModelProvider

from .memory_admission import MemoryFormationError, ground_memory_candidates
from .memory_condition_wire import ConditionSetReviewWire
from .memory_evidence import index_memory_evidence
from .memory_features import MemoryReviewFeatures
from .memory_generation import MemoryGenerationPolicy, MemoryProposalError, generate_memory
from .memory_review_wire import (
    ClaimExtractionWire,
    ExtractionWire,
    ReviewWire,
    legacy_review_schema,
)
from .memory_set_wire import FactSetReviewWire
from .structured_memory import call_structured_memory


class ModelMemoryExtractor:
    """Extract durable atomic user statements; model interpretations remain proposals."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        generation: MemoryGenerationPolicy | None = None,
        claim_modalities: bool = False,
    ) -> None:
        if claim_modalities and generation is None:
            raise ValueError("claim modalities require an explicit generation policy")
        self.provider = provider
        self._generation = generation
        self._claim_modalities = claim_modalities
        self.version = (
            f"model-memory-extractor/3:{configuration_id};{generation.version}"
            if generation
            else f"model-memory-extractor/2.1:{configuration_id}"
        )
        if claim_modalities:
            assert generation is not None
            self.version = f"model-memory-extractor/5:spans:{configuration_id};{generation.version}"

    async def extract(
        self,
        request: MemoryFormationInput,
        *,
        max_duration: float,
    ) -> MemoryExtraction:
        instructions = (
            "Extract only durable facts, preferences, constraints or explicit corrections from "
            "the focal accepted user input. Use prior documents only to resolve references; "
            "every candidate must cite the focal source plus any needed earlier source. "
            "Return zero candidates for casual talk, questions, one-off instructions, "
            "hypothetical "
            "examples or uncertain implications. Split independent claims. Do not invent details. "
            "Use author_subject only for a statement about the author, not for a statement "
            "about their project. For the bound project use input.scope.key as subject. "
            "Use consistent concise subject/predicate keys across the evidence window. "
            "value is one concise value in the original language, without commentary, "
            "alternatives, uncertainty or a sentence summarizing the change. Keep those in text. "
            "A reaffirmation repeats the original claim; an undecided alternative does not "
            "create a new condition or a second durable fact. "
            "Preserve ALL conditions (project, audience, document, exception) as qualifiers; "
            "Do not put validity dates, provenance, change status or restatements of the "
            "bound scope in qualifiers. Reuse the same words for the same conditions; don't "
            "translate condition values between inputs. "
            "Never promote task-local information to a global user preference. scope_kind must "
            "describe the scope supported by the statement; use uncertain when unclear. "
            "Each citation is a literal contiguous quote from a supplied document. start is a "
            "Unicode code-point offset; omit it for a unique quote. time_kind=unknown when the "
            "start is unspecified, current only when explicitly effective now, dated for an "
            "explicit date resolved using occurred_at and timezone. Do not infer a current "
            "start just from publication time. For current and unknown, omit valid_from and "
            "valid_to or set BOTH to null; the host resolves current. For dated, valid_from "
            "must be a UTC timestamp and valid_to is null unless an end is explicit. "
            "Use half-open intervals. Do not infer expiry. "
            "Claims about changes and corrections must retain their wording in text and quotes."
        )
        payload: dict[str, object] = {"input": request.model_dump(mode="json")}
        if self._claim_modalities:
            assert self._generation is not None
            evidence = index_memory_evidence(request)
            payload = {"input": evidence.model_input}
            # 将“材料确实报告了一个命题”和“该命题已经裁定为真”分开。
            # 竞争说法各占一条 claim;不把整段争议摘要塞进 value 或 qualifiers。
            instructions = (
                "Extract atomic reusable facts, preferences, rules and corrections from the focal "
                "accepted user input. Prior documents only resolve references. Each claim has one "
                "subject, property, concise value, applicability conditions and validity interval. "
                "Report actual competing rules separately even when nobody knows which is true. "
                "Modality: asserted=direct statement/correction; reported=an actual existing "
                "claim or record; hypothetical=possibility/suggestion; quoted=text only used in "
                "a translation/example/task. A disputed existing claim is reported, not "
                "hypothetical. "
                "Do not turn denied or withdrawn statements into positive facts. For a correction "
                "extract the corrected value and interval, retaining the full correction wording "
                "in text and citations. Conflict, uncertainty, expiry and change are annotations "
                "of these propositions, not additional standalone facts about rule status. "
                "Retention=durable means useful across tasks, INCLUDING finite validity periods; "
                "transient means a one-off task. Return zero claims when no reusable proposition "
                "is expressed. Do not invent implications. Use input.scope.key as project subject; "
                "author_subject only for author facts. Use concise stable property keys. "
                "qualifiers express when/to what the rule applies: the governed document/audience, "
                "not the document reporting a rule, the source identity, date or dispute status. "
                "Preserve all real conditions and original-language condition values. A reaffirmed "
                "general rule retains its original conditions; a new exception is a "
                "separate claim. "
                "scope_kind must match the statement's scope; never promote local facts "
                "to user scope. "
                "Each claim selects evidence indices from the supplied document spans. Include "
                "at least one focal-document span plus necessary prior spans. The host restores "
                "exact quotes and offsets: do not generate citations or count characters yourself. "
                "Select enough spans to retain negation, attribution, "
                "conditions and correction intent. "
                "time_kind=unknown if unspecified, current only if explicitly effective now, dated "
                "for explicit dates resolved to UTC using occurred_at/timezone. unknown/current "
                "omit both valid_from/valid_to. dated supplies valid_from, valid_to only for an "
                "explicit end; intervals are half-open. Never infer an expiry or start "
                "from recency."
            )

            def ground_claims(wire: ClaimExtractionWire) -> MemoryExtraction:
                try:
                    return ground_memory_candidates(request, wire.proposals(evidence.citations))
                except MemoryFormationError as exc:
                    raise MemoryProposalError("invalid_evidence") from exc

            claims, report = await generate_memory(
                self.provider,
                ClaimExtractionWire,
                "propose_memory",
                instructions,
                payload,
                max_duration,
                stage="extraction",
                policy=self._generation,
                validate=ground_claims,
            )
            return MemoryExtraction(
                schema_version=3,
                candidates=ground_claims(claims).candidates,
                modalities=tuple(c.modality for c in claims.claims),
                generation=report,
            )
        if self._generation is None:
            wire = await call_structured_memory(
                self.provider, ExtractionWire, "propose_memory", instructions, payload, max_duration
            )
            return MemoryExtraction(candidates=wire.candidates)

        def grounded(wire: ExtractionWire) -> MemoryExtraction:
            try:
                return ground_memory_candidates(
                    request, MemoryExtraction(candidates=wire.candidates)
                )
            except MemoryFormationError as exc:
                raise MemoryProposalError("invalid_evidence") from exc

        wire, report = await generate_memory(
            self.provider,
            ExtractionWire,
            "propose_memory",
            instructions,
            payload,
            max_duration,
            stage="extraction",
            policy=self._generation,
            validate=grounded,
        )
        return MemoryExtraction(
            schema_version=2, candidates=grounded(wire).candidates, generation=report
        )


class ModelMemoryReviewer:
    """A separate model call evaluates evidence and relationships; it is not a truth oracle."""

    @property
    def features(self) -> MemoryReviewFeatures:
        return MemoryReviewFeatures(self._set_reconciliation, self._semantic_conditions)

    def __init__(
        self,
        provider: ModelProvider,
        *,
        configuration_id: str,
        family_alignment: bool = False,
        generation: MemoryGenerationPolicy | None = None,
        set_reconciliation: bool = False,
        semantic_conditions: bool = False,
    ) -> None:
        if semantic_conditions and not set_reconciliation:
            raise ValueError("semantic conditions require set reconciliation")
        self._semantic_conditions = semantic_conditions
        if set_reconciliation and (generation is None or not family_alignment):
            raise ValueError("set reconciliation requires generation and family alignment")
        self.provider = provider
        self._family_alignment = family_alignment
        self._generation = generation
        self._set_reconciliation = set_reconciliation
        self.version = (
            f"model-memory-reviewer/4:family={family_alignment}:{configuration_id};{generation.version}"
            if generation
            else f"model-memory-reviewer/{'3.1' if family_alignment else '2.2'}:{configuration_id}"
        )
        if set_reconciliation:
            assert generation is not None
            self.version = f"model-memory-reviewer/6.1:sets:{configuration_id};{generation.version}"
            if semantic_conditions:
                self.version = (
                    f"model-memory-reviewer/7:conditions:{configuration_id};{generation.version}"
                )

    async def review(
        self,
        request: MemoryFormationInput,
        extraction: MemoryExtraction,
        snapshot: MemoryReconciliation,
        *,
        max_duration: float,
    ) -> MemoryReview:
        instructions = (
            "Independently verify each candidate against the full supplied source documents. "
            "A matching quote does not prove entailment. Reject inventions, negation errors, "
            "hypothetical claims, missing conditions, overbroad scope or unsupported dates. "
            "Review every candidate exactly once using its zero-based index. Existing memory "
            "is a complete bounded snapshot, including past, future and unknown-time facts. "
            "Determine relation: new has no same-key overlapping fact; equivalent repeats the "
            "same value; change explicitly changes the value from a supported later time; "
            "correction explicitly says the earlier statement was wrong (not simply a later "
            "change); conflict gives incompatible overlapping evidence without resolving it; "
            "unresolved lacks enough evidence to decide; irrelevant should not be retained. "
            "Only equivalent/change/correction/conflict supply target_id from the snapshot. "
            "Target must denote the same subject, predicate and conditions, even if words "
            "differ. A conditional exception must not overwrite the general fact. If multiple "
            "targets would need joint revision, return unresolved. New evidence is not "
            "automatically more authoritative. Never choose a winner merely by recency. "
            "For every target relationship, provide an alignment "
            "binding candidate_index, target_id and target_revision from the snapshot. "
            "Each match has candidate_path, target_path and literal citations from the focal "
            "source (already cited by the candidate). Paths are /subject, /predicate, /value "
            "or /qualifiers/<JSON-pointer-escaped-key>. Match /subject only to /subject. "
            "Cover /predicate and EVERY qualifier on BOTH sides; repeated paths are allowed "
            "when a predicate already includes a condition, but do not erase a real condition. "
            "Supply /value to /value only for equivalent values, including translations. "
            "Do not align genuinely different applicability (e.g. a general rule and an "
            "exception). If you cannot establish all correspondences, return unresolved. "
            "Several candidates may target one fact when they explicitly partition its whole "
            "valid interval: an unchanged prefix is equivalent, and a changed later segment "
            "is change. The host validates the complete partition atomically. "
            + (
                "Return schema_version=3. families is distinct from alignments: a NEW fact may "
                "join an existing subject/property family despite different conditions or time. "
                "For each such fact provide candidate_index, target_id, target_revision and "
                "exactly two matches (/subject to /subject, /predicate to /predicate), with "
                "literal focal citations as above. Choose an existing semantic assertion as "
                "anchor; its value is not a winner. Compare meaning, not spelling or language. "
                "Use families for conditional exceptions and disjoint validity intervals, even "
                "when their subject/predicate spelling already matches. Keep all candidate "
                "qualifiers, value and validity unchanged. Only map the same entity and property; "
                "related topics alone do not establish a family. If a predicate embeds a condition "
                "not explicitly retained in candidate qualifiers, return unresolved. Do not hide "
                "a competing same-condition value as new: use conflict with a target alignment. "
                "No family link authorizes correction or overwriting. If no family exists, omit "
                "the link; if membership is ambiguous return unresolved."
                if self._family_alignment
                else "Return schema_version=2; omit families."
            )
        )
        payload: dict[str, object] = {
            "input": request.model_dump(mode="json"),
            "proposals": {
                "schema_version": 1,
                "candidates": [c.model_dump(mode="json") for c in extraction.candidates],
            },
            "memory": snapshot.model_dump(mode="json"),
        }
        if self._set_reconciliation:
            assert self._generation is not None
            payload["claim_modalities"] = extraction.modalities
            payload["proposals"] = {
                "candidates": [
                    dict(index=i, **candidate.model_dump(mode="json"))
                    for i, candidate in enumerate(extraction.candidates)
                ]
            }
            payload["memory"] = {
                "head": snapshot.head,
                "versions": [
                    dict(index=i, **v.model_dump(mode="json"))
                    for i, v in enumerate(snapshot.versions)
                ],
            }
            instructions = (
                "Reconcile each candidate against the full evidence and memory snapshot. Return "
                "one decision per candidate index. verdict checks factual support, scope, "
                "conditions "
                "and dates. An actual reported rule is supported as a proposition even "
                "when disputed. "
                "Hypothetical/quoted statements are irrelevant. Relations: new=no overlapping same-"
                "condition fact; equivalent=same value; conflict=incompatible overlapping value; "
                "change=explicit later change; correction=explicit earlier error; "
                "unresolved=unclear; "
                "irrelevant=not reusable. Recency alone never settles competing evidence. "
                "anchor is the INDEX of a same-condition target for equivalent/conflict/change/"
                "correction. For new, optionally anchor a same-entity/property family "
                "with different "
                "conditions/time. Omit anchor otherwise and leave mapping/resolves EMPTY: there is "
                "no stored counterpart to map a standalone new fact onto. Only WITH an anchor, "
                "mapping pairs candidate and target paths: /subject, /predicate, /value, "
                "/qualifiers/<JSON-pointer-escaped-key>. Pair /subject with /subject. "
                "For new-family map only subject and predicate. "
                + (
                    "For other target relations map only subject and the full predicate meaning; "
                    "do not map qualifier paths. Do not claim predicate equivalence if it hides "
                    "a different condition. "
                    if self._semantic_conditions
                    else "For other target relations cover predicate and EVERY qualifier "
                    "on BOTH sides; "
                    "a condition embedded in a predicate may map to an explicit qualifier. "
                )
                + "Never equate genuinely "
                "different conditions. For equivalent values add /value to /value if "
                "wording differs. "
                "citations are indices in THIS candidate's citations supporting these "
                "correspondences; "
                "include focal evidence. Do not output quotes, target IDs or revisions. "
                "When evidence explicitly settles the whole conflicting set in the candidate's "
                "interval, use correction and resolves=[ALL affected memory INDICES], including "
                "targets already agreeing with the corrected value. Exclude different conditions "
                "and disjoint intervals. Without explicit settlement leave resolves empty. Partial "
                "time corrections are allowed: the host retains unaffected intervals and computes "
                "all fragments. Do not generate fragments or choose a winner for "
                "unresolved conflict."
            )
            if self._semantic_conditions:
                # 条件图替代完整 qualifier 路径对应;没有写入领域例子或同义词表。
                instructions += (
                    " For each anchor, compare applicability conditions separately "
                    "in condition_links: "
                    "[left SIDE, left qualifier KEY, right SIDE, right qualifier KEY, relation]. "
                    "SIDE is candidate or target; within-side links are allowed for redundant "
                    "conditions. Each atom means the complete role-and-value restriction in its "
                    "source context. equivalent means mutual entailment; entails means the left "
                    "restriction implies the right restriction; disjoint means both cannot hold "
                    "at once. Omit uncertain pairs. "
                    "Do not confuse semantic similarity with entailment. Preserve negation and "
                    "distinguish attribution from applicability; never drop an existing qualifier. "
                    "Supply supported links needed for coverage, including repeated "
                    "or redundant restrictions; the host computes closure, mutual coverage and "
                    "contradictions. Identical keys and values get automatic identity links. "
                    "Empty condition sets express no explicit restriction. "
                    "No anchor means no links. "
                    "mapping still establishes subject/predicate identity and optional value "
                    "equivalence; citations support both mapping and condition links."
                )
                wire, report = await generate_memory(
                    self.provider,
                    ConditionSetReviewWire,
                    "review_memory",
                    instructions,
                    payload,
                    max_duration,
                    stage="review",
                    policy=self._generation,
                    validate=lambda w: w.interpret(extraction, snapshot),
                )
                interpreted, conditions = wire.interpret(extraction, snapshot)
                result = interpreted.expand(extraction, families=True)
                return MemoryReview.model_validate(
                    {
                        **result.model_dump(),
                        "schema_version": 6,
                        "generation": report,
                        "resolutions": interpreted.expand_resolutions(extraction),
                        "conditions": conditions,
                    }
                )
            fact_wire, report = await generate_memory(
                self.provider,
                FactSetReviewWire,
                "review_memory",
                instructions,
                payload,
                max_duration,
                stage="review",
                policy=self._generation,
                validate=lambda w: w.interpret(extraction, snapshot),
            )
            interpreted = fact_wire.interpret(extraction, snapshot)
            result = interpreted.expand(extraction, families=True)
            return MemoryReview.model_validate(
                {
                    **result.model_dump(),
                    "schema_version": 5,
                    "generation": report,
                    "resolutions": interpreted.expand_resolutions(extraction),
                }
            )
        if self._generation is not None:
            # 模型只选择候选中已通过 grounding 的引用,不再反复抄写长原文。
            # wire schema_version=1 与持久 Review v4 是不同边界。
            instructions = instructions.replace(
                "Return schema_version=3.", "Return schema_version=1."
            )
            instructions = instructions.replace(
                "Return schema_version=2; omit families.", "Return schema_version=1; omit families."
            )
            instructions = instructions.replace(
                "literal citations from the focal source (already cited by the candidate)",
                "indices into the candidate's grounded citations",
            ).replace("literal focal citations as above", "candidate citation references as above")
            instructions += (
                " For this compact wire protocol, each match.citations is an array of zero-based "
                "integer indices into THAT candidate's citations, not quote objects. All original "
                "quotes remain in proposals; the host expands the selected references exactly. "
                "Select a reference covering the focal source where required. Never invent indices."
            )
            wire, report = await generate_memory(
                self.provider,
                ReviewWire,
                "review_memory",
                instructions,
                payload,
                max_duration,
                stage="review",
                policy=self._generation,
                validate=lambda w: w.expand(extraction, families=self._family_alignment),
            )
            result = wire.expand(extraction, families=self._family_alignment)
            return MemoryReview.model_validate(
                {
                    **result.model_dump(),
                    "schema_version": 4,
                    "generation": report,
                }
            )
        result = await call_structured_memory(
            self.provider,
            MemoryReview,
            "review_memory",
            instructions,
            payload,
            max_duration,
            visible_schema=legacy_review_schema(families=self._family_alignment),
        )
        if result.schema_version != (3 if self._family_alignment else 2):
            raise MemoryFormationError("memory reviewer returned a different configured protocol")
        return result
