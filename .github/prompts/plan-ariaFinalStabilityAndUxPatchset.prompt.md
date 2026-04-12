## Plan: ARIA Final Stability and UX Patchset

Implement conversation-aware intent parsing (rules + retraining), merge wakeword advanced controls into structured UI, rebrand visible text/assistant identity/voice labels, force LLM worker mode with crash/status telemetry and speed tuning, and expand the intent identifier panel with full intent catalog plus top-3 scoring diagnostics. Keep internal storage keys and path constants unchanged for migration safety.

**Steps**
1. Phase 0 - Baseline and safety gates.
- Capture baseline behavior with quick smoke scripts around JarvisPipeline.initialize/process_text and analyze_image_file.
- Export a before/after checklist for all requested areas: intent phrasing, wakeword UI, branding/voices, LLM/vision, and intent panel.
2. Phase 1 - Conversational intent understanding (core parser path). Depends on step 1.
- Add a conversational phrase normalizer module (prefix/suffix stripping and polite-filler normalization) and call it in core pipeline before classification and entity extraction.
- Expand entity extraction patterns to handle full-sentence command forms and compound phrasing: “do me a favor…”, “can you please…”, implied app targets, direction aliases, and number-without-percent for volume/brightness.
- Keep fallback behavior safe: if normalization fails, use current cleaned text.
3. Phase 2 - Intent data coverage and retraining. Depends on step 2.
- Extend dataset augmentation templates with polite prefixes/suffixes and natural spoken variants across all existing intents.
- Regenerate intents dataset and retrain primary + fallback intent models so classifier confidence improves on natural phrasing.
- Compare pre/post intent metrics and keep threshold tuning conservative to avoid over-triggering actionable intents.
4. Phase 3 - Intent Identifier panel expansion (compact UX). Depends on step 2 for telemetry; can run in parallel with step 3 retraining.
- Extend pipeline-to-UI intent telemetry to include top-3 candidate intents, all_scores summary, runtime/provider, and latency.
- Update sidebar intent panel to remain compact while adding:
- full supported intent catalog with one-line descriptions,
- top-3 live candidates with confidence,
- per-intent example hints (tooltip/hover),
- runtime info (model/provider/latency).
- Preserve current model selector behavior.
5. Phase 4 - Wakeword settings merge (no raw JSON editing). Parallel with steps 3 and 4.
- Remove direct raw JSON editor from settings dialog and keep all advanced options as validated UI controls in one unified wakeword section.
- Ensure every advanced wakeword option currently in config remains editable via UI and still persists to wakeword.json through existing save path.
- Add clear apply/reinitialize feedback so users know when detector reinit is in progress/complete.
6. Phase 5 - Branding and voice profile UX updates. Parallel with step 5.
- Replace visible phrase “Adaptive Reasoning & Intelligence Architecture” with “Just A Rather Very Intelligent Assistant” in UI/docs and assistant identity prompts.
- Update voice profile labels in UI: male label becomes JARVIS, female label becomes FRIDAY, while internal saved keys remain male/female.
- Add dynamic profile description text in settings:
- FRIDAY: “Female Replacement Intelligent Digital Assistant Youth”
- JARVIS: “Just A Rather Very Intelligent Assistant”
- Update default wake phrases/identity strings to match selected branding scope while preserving backward compatibility aliases where needed.
7. Phase 6 - LLM/vision stability, crash handling, and speed. Depends on step 1 baseline; parallel with steps 5–6 where file overlap allows.
- Set subprocess worker mode as GUI default (stability-first) and avoid in-process llama initialization path in GUI runtime.
- Add robust worker lifecycle/status handling: startup, ready, error/crash, restart attempts; ensure router reuses bridge instead of churn.
- Add visible LLM status indicator under the orb (initializing, ready, degraded, crashed/restarting) with user-friendly messaging.
- Improve responsiveness by prewarming worker after pipeline init, tuning token/context defaults for faster average response, and avoiding unnecessary model/runtime restarts.
- Ensure vision requests degrade gracefully with explicit, actionable error text if multimodal runtime is unavailable.
8. Phase 7 - UI responsiveness polish. Depends on step 7.
- Tighten long-task hints and background-task state transitions so send/mic controls reflect busy/ready accurately during LLM and vision operations.
- Prevent UI-thread blocking in any new status/update paths introduced above.
9. Phase 8 - Verification and regression sweep. Depends on all prior steps.
- Run import and smoke checks for pipeline initialization, natural-language command handling, wakeword settings persistence, LLM text response, and image analysis flow.
- Run focused command-set manual tests using polite/full-sentence prompts for each intent family.
- Reopen settings and verify wakeword+voice/branding changes persist across restart.
- Validate intent panel compactness on desktop window default size.

**Relevant files**
- c:\PROJECTS\A_R_I_A\core\pipeline.py — preprocess/intent flow, intent signal payload, wakeword settings apply, LLM prewarm/status hooks.
- c:\PROJECTS\A_R_I_A\nlp\preprocessor.py — normalization entry point for conversation-style command cleanup.
- c:\PROJECTS\A_R_I_A\nlp\entity_extractor.py — robust slot extraction for natural sentence commands and numeric/direction variants.
- c:\PROJECTS\A_R_I_A\nlp\intent_classifier.py — confidence and score handling reused for top-3 UI telemetry.
- c:\PROJECTS\A_R_I_A\ml\dataset.py — expanded polite/full-sentence augmentation templates.
- c:\PROJECTS\A_R_I_A\ui\widgets\sidebar.py — compact intent panel with full intent catalog + top-3 confidence + runtime info.
- c:\PROJECTS\A_R_I_A\ui\main_window.py — settings dialog merge, voice labels/descriptions, LLM status under orb.
- c:\PROJECTS\A_R_I_A\speech\wakeword_config.py — advanced wakeword fields/defaults persisted to JSON.
- c:\PROJECTS\A_R_I_A\speech\tts.py — profile-label mapping support and profile metadata exposure.
- c:\PROJECTS\A_R_I_A\llm\qwen_bridge.py — force worker default, worker lifecycle hardening, speed tuning knobs.
- c:\PROJECTS\A_R_I_A\llm\qwen_worker.py — worker startup/response resilience and clear structured errors.
- c:\PROJECTS\A_R_I_A\nlp\router.py — LLM readiness/vision routing and fallback messaging.
- c:\PROJECTS\A_R_I_A\README.md — architecture phrase replacement and branding consistency updates.
- c:\PROJECTS\A_R_I_A\core\config.py — only if branding text constants are surfaced; no path migration unless explicitly approved.

**Verification**
1. Run python __copilot_import_check.py to validate import safety after edits.
2. Run a focused pipeline smoke script covering initialize(), process_text() with polite/full-sentence variants for each intent family, and analyze_image_file() with an attached image.
3. Regenerate and retrain intent data/models: python ml/dataset.py, python ml/train_ml.py, python ml/train_distilbert.py.
4. Compare classification quality and latency with python ml/evaluate.py.
5. Manual UI checks:
- settings dialog has unified wakeword controls (no raw JSON editing required),
- voice labels/descriptions switch correctly (FRIDAY/JARVIS),
- LLM status indicator below orb updates on ready/crash/restart,
- intent panel shows compact catalog + top-3 + runtime info.
6. Manual runtime checks:
- conversational commands like “can you please launch chrome” and “do me a favor and set brightness to 60” execute correctly,
- local LLM replies and image analysis both return without silent failures.

**Decisions**
- Intent strategy: rule-based conversational normalization plus model retraining.
- Branding scope: visible text plus assistant identity and wake phrase defaults.
- Voice key strategy: keep internal stored keys female/male; change only labels/descriptions in UI.
- Intent panel content: include all intents, top-3 confidence candidates, example hints, and runtime info.
- LLM mode: worker subprocess is the default mode.
- LLM crash UX: add visible user-facing status indication below orb.
- Speed target: prioritize perceived responsiveness (startup prewarm, lower average latency, non-blocking UI updates) while preserving stability.

**Further Considerations**
1. Optional: keep both FRIDAY and JARVIS as accepted wake aliases by default to avoid breaking existing user habits after branding updates.
2. Optional: expose an advanced LLM performance profile setting (balanced vs fast) for max_tokens/context tuning without code changes.
3. Out of scope unless later requested: renaming internal class names, module names, or APPDATA path identifiers that would require migration logic.
