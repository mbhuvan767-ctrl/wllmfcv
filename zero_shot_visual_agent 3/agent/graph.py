"""The zero-shot visual agent (LangGraph).

    plan -> tools -> reconcile --(unchecked claims?)--> verify -> reconcile -> compose -> END
                         \\_______________________________________________/

* plan       rule-based router: picks which tools answer the question
* tools      runs BLIP VQA / OWL-ViT / CLIP / ResNet-50 / safety screening (each isolated, errors captured)
* reconcile  turns the VQA answer into checkable claims and cross-checks them against detector output
* verify     one bounded extra detector pass on claims nobody has checked yet
* compose    builds the final answer, evidence list, uncertainty notes and agreement level
"""
import operator
import time
from collections import Counter
from typing import Annotated, Any, Callable, Optional, TypedDict

from . import config as C
from .safety import analyze_safety
from .text_utils import count_targets, extract_terms, is_yes_no, parse_number, prompt_for
from .utils import softmax

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    LANGGRAPH_AVAILABLE = False

TOOL_ORDER = ["vqa", "detection", "clip", "cnn", "safety"]
TOOL_MODELS = {"vqa": ["vqa"], "detection": ["owl"], "clip": ["clip"], "cnn": ["cnn"],
               "safety": ["clip", "owl"]}
TOOL_TITLES = {"vqa": "BLIP VQA", "detection": "OWL-ViT detection", "clip": "CLIP zero-shot",
               "cnn": "ResNet-50", "safety": "Safety screening"}
CONFLICT_SCORE = 0.30


class AgentState(TypedDict, total=False):
    image: Any
    question: str
    options: dict
    plan: list
    plan_reasons: dict
    tool_results: dict
    claims: list
    evidence: list
    uncertainty: list
    agreement: str
    verification_count: int
    needs_verification: bool
    summary: str
    answer: str
    final: dict
    trace: Annotated[list, operator.add]


def _merge(state: dict, update: dict) -> dict:
    """Apply a node update to the running state (trace is appended, others overwritten)."""
    state = dict(state)
    for k, v in (update or {}).items():
        state[k] = (state.get(k, []) + v) if k == "trace" else v
    return state


def _step(name: str, detail: str, t0: float) -> list:
    return [{"step": name, "detail": detail, "seconds": round(time.time() - t0, 2)}]


def _matches(term: str, label: str) -> bool:
    return term == label or term in label or label in term


class VisualAgent:
    def __init__(self, hub):
        self.hub = hub
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        self.backend = "LangGraph" if self.graph is not None else "sequential fallback"

    # ================================================================ nodes
    def plan_tools(self, state: AgentState) -> dict:
        t0 = time.time()
        q = state["question"].lower()
        opts = state.get("options", {})
        plan: list = []
        reasons: dict = {}

        def add(tool, why):
            if tool not in reasons:
                plan.append(tool)
                reasons[tool] = why

        forced = opts.get("forced_tools")
        if forced:
            for t in forced:
                add(t, "selected manually in the sidebar")
        else:
            add("vqa", "answers the natural-language question directly")
            if opts.get("mode") == "safety" or any(t in q for t in C.SAFETY_TERMS):
                add("safety", "safety-related question or Safety mode")
            if count_targets(q, C.KNOWN_PHRASES):
                add("detection", "'how many' questions are answered by counting detections")
            if any(k in q for k in C.OBJECT_KEYWORDS):
                add("detection", "question asks about objects or their locations")
            if any(k in q for k in C.CLASSIFY_KEYWORDS):
                add("clip", "question asks for a scene/category")
                add("cnn", "supervised ImageNet baseline for the same question")
            if opts.get("custom_labels"):
                add("detection", "custom labels were supplied")
                add("clip", "custom labels were supplied (zero-shot classification)")
        if "safety" in plan:
            add("detection", "safety screening needs person / vehicle / PPE detections")
        ordered = [t for t in TOOL_ORDER if t in reasons]
        return {"plan": ordered, "plan_reasons": reasons, "tool_results": {}, "verification_count": 0,
                "trace": _step("plan", "tools: " + (", ".join(ordered) or "none"), t0)}

    def _detection_labels(self, state) -> list:
        opts = state.get("options", {})
        priority = list(opts.get("custom_labels") or [])
        priority += count_targets(state["question"], C.KNOWN_PHRASES)
        priority += extract_terms(state["question"], C.KNOWN_PHRASES)
        labels = priority + C.CORE_DETECTION_LABELS
        if "safety" in state.get("plan", []):
            labels += C.WORKSITE_DETECTION_LABELS
        return list(dict.fromkeys(l.lower().strip() for l in labels if l.strip()))[:C.MAX_DETECTION_LABELS]

    def _filter_min_score(self, dets):
        return [d for d in dets if d["score"] >= C.LABEL_MIN_SCORE.get(d["label"], 0.0)]

    def _run_vqa(self, image, state, results):
        return {"answer": self.hub.vqa(image, state["question"])}

    def _run_detection(self, image, state, results):
        labels = self._detection_labels(state)
        thr = state.get("options", {}).get("threshold", 0.12)
        dets = self._filter_min_score(self.hub.detect(image, labels, threshold=thr))
        return {"labels_queried": labels, "detections": dets,
                "counts": dict(Counter(d["label"] for d in dets))}

    def _run_clip(self, image, state, results):
        custom = state.get("options", {}).get("custom_labels") or []
        names = custom or C.GENERAL_SCENE_LABELS
        logits = self.hub.clip_logits([image], [prompt_for(n) for n in names])[0]
        probs = softmax(logits)
        order = probs.argsort()[::-1][:5]
        return {"label_set": "custom labels" if custom else "general scene labels",
                "rows": [{"label": names[i], "score": round(float(probs[i]), 4)} for i in order]}

    def _run_cnn(self, image, state, results):
        return {"rows": self.hub.cnn_topk(image, 5)}

    def _run_safety(self, image, state, results):
        det = results.get("detection")
        dets = det["detections"] if isinstance(det, dict) and "detections" in det else None
        return analyze_safety(self.hub, image, dets, state.get("options", {}).get("threshold", 0.12))

    def execute_tools(self, state: AgentState) -> dict:
        t0 = time.time()
        image = state["image"]
        results = dict(state.get("tool_results", {}))
        for tool in state.get("plan", []):
            if tool in results:
                continue
            missing = [m for m in TOOL_MODELS[tool] if not self.hub.ensure(m)]
            if missing:
                results[tool] = {"error": f"model '{missing[0]}' unavailable ({self.hub.error(missing[0])})"}
                continue
            try:
                results[tool] = getattr(self, f"_run_{tool}")(image, state, results)
            except Exception as e:
                results[tool] = {"error": f"{type(e).__name__}: {e}"}
        failed = [t for t, r in results.items() if isinstance(r, dict) and r.get("error")]
        detail = f"ran {len(results) - len(failed)} tool(s)" + (f", {len(failed)} failed: {', '.join(failed)}" if failed else "")
        return {"tool_results": results, "trace": _step("tools", detail, t0)}

    # -------------------------------------------------------------- reconcile
    @staticmethod
    def _detector_pool(results):
        pool, queried = {}, set()
        for key in ("detection", "verification_detection"):
            block = results.get(key)
            if not isinstance(block, dict) or block.get("error"):
                continue
            queried.update(block.get("labels_queried", []))
            for d in block.get("detections", []):
                pool[d["label"]] = max(pool.get(d["label"], 0.0), d["score"])
        return pool, queried

    def _build_claims(self, state, results):
        vqa = results.get("vqa", {})
        answer = (vqa.get("answer") or "").strip() if isinstance(vqa, dict) else ""
        if not answer:
            return []
        if is_yes_no(answer):
            terms = extract_terms(state["question"], C.KNOWN_PHRASES)[:3]
            polarity = "present" if answer.lower().startswith("yes") else "absent"
        else:
            terms = extract_terms(answer, C.KNOWN_PHRASES)[:4]
            polarity = "present"
        return [{"term": t, "polarity": polarity} for t in terms]

    def reconcile_results(self, state: AgentState) -> dict:
        t0 = time.time()
        results = state.get("tool_results", {})
        opts = state.get("options", {})
        evidence, uncertainty = [], []

        for tool, res in results.items():
            if isinstance(res, dict) and res.get("error"):
                uncertainty.append(f"{TOOL_TITLES.get(tool, tool)} failed: {res['error']}")

        vqa = results.get("vqa", {})
        answer = (vqa.get("answer") or "").strip() if isinstance(vqa, dict) else ""
        if answer:
            evidence.append({"source": "BLIP VQA", "finding": answer})

        pool, queried = self._detector_pool(results)
        det = results.get("detection")
        if isinstance(det, dict) and not det.get("error"):
            if det.get("detections"):
                best = {}
                for d in det["detections"]:
                    best[d["label"]] = max(best.get(d["label"], 0), d["score"])
                evidence.append({"source": "OWL-ViT", "finding": "Detected: " + ", ".join(
                    f"{n}x {l} (best {best[l]:.2f})" for l, n in det["counts"].items())})
            else:
                uncertainty.append("OWL-ViT returned no objects above the threshold.")

        clip = results.get("clip")
        if isinstance(clip, dict) and clip.get("rows"):
            evidence.append({"source": "CLIP", "finding": f"Top {clip['label_set']}: " + ", ".join(
                f"{r['label']} ({r['score']:.2f})" for r in clip["rows"][:3])})
            uncertainty.append("CLIP scores are relative to the candidate labels, not calibrated probabilities.")
        cnn = results.get("cnn")
        if isinstance(cnn, dict) and cnn.get("rows"):
            evidence.append({"source": "ResNet-50", "finding": "ImageNet candidates: " + ", ".join(
                f"{r['class']} ({r['score']:.2f})" for r in cnn["rows"][:3])})
            uncertainty.append("ResNet-50 is limited to the 1000 ImageNet classes.")
        safety = results.get("safety")
        if isinstance(safety, dict) and not safety.get("error"):
            flagged = [f for f in safety["findings"] if f["severity"] in ("medium", "low")]
            evidence.append({"source": "Safety screening", "finding": (
                f"{len(flagged)} potential indicator(s) flagged" if flagged else "no obvious indicators flagged")})
            uncertainty.append(safety["warning"])

        # ---- claim cross-check: VQA vs detector
        claims = self._build_claims(state, results)
        for c in claims:
            s = max((v for l, v in pool.items() if _matches(c["term"], l)), default=0.0)
            checked = any(_matches(c["term"], l) for l in queried)
            if c["polarity"] == "present":
                c["status"] = "supported" if s > 0 else ("not_corroborated" if checked else "unchecked")
            else:
                c["status"] = "conflict" if s >= CONFLICT_SCORE else ("consistent" if checked else "unchecked")
            c["detector_score"] = round(s, 3)
            if c["status"] == "not_corroborated":
                uncertainty.append(f"BLIP mentioned '{c['term']}' but the detector did not find it "
                                   "(detector miss or VQA hallucination).")
            elif c["status"] == "conflict":
                uncertainty.append(f"BLIP implies '{c['term']}' is absent but the detector found it (score {s:.2f}).")

        # ---- count questions: compare BLIP number with detector count
        targets = count_targets(state["question"], C.KNOWN_PHRASES)
        count_mismatch = False
        if targets and isinstance(det, dict) and det.get("counts") is not None:
            n_det = sum(n for l, n in det["counts"].items() if any(_matches(t, l) for t in targets))
            n_vqa = parse_number(answer) if answer else None
            if n_vqa is not None and n_vqa != n_det:
                count_mismatch = True
                uncertainty.append(f"BLIP answered {n_vqa} but the detector counted {n_det}; "
                                   "VQA models are weak at counting.")

        unchecked = [c for c in claims if c["status"] == "unchecked"]
        needs_verification = False
        if unchecked and opts.get("verify", True) and state.get("verification_count", 0) == 0:
            if self.hub.ensure("owl"):
                needs_verification = True
            else:
                uncertainty.append(f"Could not cross-check VQA claims: detector unavailable ({self.hub.error('owl')}).")

        bad = count_mismatch or any(c["status"] in ("conflict", "not_corroborated") for c in claims)
        good = any(c["status"] in ("supported", "consistent") for c in claims)
        agreement = "mixed" if bad else ("consistent" if good else "unverified")
        n_claims = len(claims)
        detail = (f"{n_claims} claim(s) checked, agreement={agreement}"
                  + ("; verification requested" if needs_verification else ""))
        return {"evidence": evidence, "uncertainty": list(dict.fromkeys(uncertainty)), "claims": claims,
                "agreement": agreement, "needs_verification": needs_verification,
                "trace": _step("reconcile", detail, t0)}

    def verify_again(self, state: AgentState) -> dict:
        t0 = time.time()
        results = dict(state.get("tool_results", {}))
        terms = [c["term"] for c in state.get("claims", []) if c["status"] == "unchecked"]
        try:
            thr = min(state.get("options", {}).get("threshold", 0.12), 0.10)
            dets = self._filter_min_score(self.hub.detect(state["image"], terms, threshold=thr))
            results["verification_detection"] = {"labels_queried": terms, "detections": dets,
                                                 "counts": dict(Counter(d["label"] for d in dets))}
        except Exception as e:
            results["verification_detection"] = {"error": f"{type(e).__name__}: {e}", "labels_queried": terms}
        return {"tool_results": results, "verification_count": state.get("verification_count", 0) + 1,
                "trace": _step("verify", "re-checked with detector: " + ", ".join(terms), t0)}

    # ---------------------------------------------------------------- compose
    def compose_answer(self, state: AgentState) -> dict:
        t0 = time.time()
        q = state["question"]
        results = state.get("tool_results", {})
        vqa = results.get("vqa", {})
        answer = (vqa.get("answer") or "").strip() if isinstance(vqa, dict) else ""
        det = results.get("detection") if isinstance(results.get("detection"), dict) else {}
        safety = results.get("safety") if isinstance(results.get("safety"), dict) else {}

        lines = []
        targets = count_targets(q, C.KNOWN_PHRASES)
        if targets and det.get("counts") is not None:
            counts = {t: sum(n for l, n in det["counts"].items() if _matches(t, l)) for t in targets}
            lines.append("**Detector count:** " + ", ".join(f"{n} x {t}" for t, n in counts.items()) + ".")
        if answer:
            lines.append(f"**BLIP answer:** {answer}")
        else:
            lines.append("**No VQA answer was produced** (see uncertainty notes).")
        if det.get("counts"):
            lines.append("**Objects found:** " + ", ".join(f"{n} x {l}" for l, n in det["counts"].items()) + ".")
        if safety and not safety.get("error"):
            flagged = [f for f in safety["findings"] if f["severity"] in ("medium", "low")]
            lines.append(f"**Safety screening:** {len(flagged)} potential indicator(s) flagged." if flagged
                         else "**Safety screening:** no obvious indicators flagged.")
        summary = "\n\n".join(lines)

        parts = [f"Question: {q}", f"Answer: {answer or 'n/a'}", "", "Evidence gathered:"]
        parts += [f"- {e['source']}: {e['finding']}" for e in state.get("evidence", [])] or ["- none"]
        if state.get("claims"):
            parts += ["", "Claim check:"] + [f"- {c['term']} ({c['polarity']}): {c['status']}" for c in state["claims"]]
        if state.get("uncertainty"):
            parts += ["", "Uncertainty / review notes:"] + [f"- {u}" for u in state["uncertainty"]]
        full = "\n".join(parts)

        final = {"question": q, "answer": full, "summary": summary, "agreement": state.get("agreement"),
                 "plan": state.get("plan", []), "plan_reasons": state.get("plan_reasons", {}),
                 "tool_results": results, "claims": state.get("claims", []),
                 "evidence": state.get("evidence", []), "uncertainty": state.get("uncertainty", [])}
        return {"answer": full, "summary": summary, "final": final,
                "trace": _step("compose", "final answer built", t0)}

    # ----------------------------------------------------------------- graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self.plan_tools)
        g.add_node("tools", self.execute_tools)
        g.add_node("reconcile", self.reconcile_results)
        g.add_node("verify", self.verify_again)
        g.add_node("compose", self.compose_answer)
        g.set_entry_point("plan")
        g.add_edge("plan", "tools")
        g.add_edge("tools", "reconcile")
        g.add_conditional_edges("reconcile", lambda s: "verify" if s.get("needs_verification") else "compose",
                                {"verify": "verify", "compose": "compose"})
        g.add_edge("verify", "reconcile")
        g.add_edge("compose", END)
        return g.compile()

    # ------------------------------------------------------------------- run
    def run(self, image, question: str, options: Optional[dict] = None,
            on_step: Optional[Callable] = None) -> dict:
        """Run the agent. `on_step(node_name, update, state)` is called after each node."""
        t0 = time.time()
        state: dict = {"image": image.convert("RGB"),
                       "question": (question or "").strip() or "Describe what is in this image.",
                       "options": options or {}, "verification_count": 0, "tool_results": {}, "trace": []}

        def apply(node, update):
            nonlocal state
            state = _merge(state, update)
            if on_step:
                on_step(node, update, state)

        if self.graph is not None:
            for chunk in self.graph.stream(state, stream_mode="updates"):
                for node, update in chunk.items():
                    apply(node, update)
        else:  # same logic without LangGraph
            apply("plan", self.plan_tools(state))
            apply("tools", self.execute_tools(state))
            apply("reconcile", self.reconcile_results(state))
            if state.get("needs_verification"):
                apply("verify", self.verify_again(state))
                apply("reconcile", self.reconcile_results(state))
            apply("compose", self.compose_answer(state))

        final = dict(state["final"])
        final["trace"] = state["trace"]
        final["backend"] = self.backend
        final["elapsed_seconds"] = round(time.time() - t0, 2)
        return final
