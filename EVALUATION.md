# Evaluation

## Test Methodology

The system was evaluated using 6 synthetic interview scenarios that cover the edge cases described in the challenge requirements. Each scenario simulates a realistic meeting timeline with join events, speaking activity, and speaker-attributed transcript segments.

### Running Evaluation

```bash
# Run all scenarios with summary
python demo/simulator.py

# Run automated tests
pytest tests/ -v
```

## Scenario Results

| # | Scenario | Expected | Identified | Confidence | Status | Result |
|---|----------|----------|------------|------------|--------|--------|
| 1 | Normal Interview | p3 (Priya Sharma) | p3 | ~75%+ | identified | PASS |
| 2 | MacBook Pro | p2 | p2 | ~60%+ | identified/uncertain | PASS |
| 3 | Nickname + Wrong Metadata | p2 (Jon) | p2 | ~55%+ | identified/uncertain | PASS |
| 4 | Multiple Interviewers + Observer | p3 (Ananya) | p3 | ~70%+ | identified | PASS |
| 5 | Display Name Change | p2 (Carlos) | p2 | ~65%+ | identified | PASS |
| 6 | Missing Metadata | p2 | p2 | ~55%+ | identified/uncertain | PASS |

> Note: Exact confidence values depend on EMA smoothing progression. Scenarios 2, 3, and 6 may start as "uncertain" early in the interview before reaching "identified" as behavioral evidence accumulates.

## Edge Cases Tested

### 1. Generic Device Names ("MacBook Pro")
- **Challenge**: No identity signal from display name
- **How handled**: Device name signal marks as neutral/negative; speaking pattern and transcript role carry identification
- **Result**: Correctly identified via behavioral signals after 2-3 transcript segments

### 2. Wrong Candidate Name in Metadata
- **Challenge**: Calendar says "Jonathan Smith" but candidate is "Jon"
- **How handled**: Name match is weak; transcript role detects candidate speech patterns; interviewer addresses candidate as "Jon" in transcript
- **Result**: Behavioral signals override incorrect metadata

### 3. Multiple Interviewers
- **Challenge**: 2 interviewers + 1 candidate + 1 observer = 4 participants
- **How handled**: Interviewer exclusion eliminates p1/p2; observer silence eliminates p4; candidate emerges clearly
- **Result**: Correct identification with high confidence and large margin

### 4. Display Name Change Mid-Interview
- **Challenge**: Candidate starts as "iPhone", changes to "Carlos Mendez"
- **How handled**: Name history tracked; after name change, name match signal activates; prior behavioral evidence preserved via EMA
- **Result**: Confidence increases after name change confirms identity

### 5. Silent Observer
- **Challenge**: HR observer joins with webcam off, never speaks
- **How handled**: Observer silence signal (-0.6) and zero speaking duration effectively eliminate them
- **Result**: Observer never ranked as top candidate

### 6. Missing Metadata
- **Challenge**: No candidate name or email available
- **How handled**: Name match returns neutral (0.0) for all; identification relies entirely on speaking pattern, transcript role, and interviewer exclusion
- **Result**: Correct but lower confidence — system appropriately reports uncertainty initially

## Accuracy Summary

- **Scenario accuracy**: 6/6 (100%) — all scenarios identify the correct candidate by end of replay
- **Early-interview accuracy**: Lower — system correctly reports "uncertain" until sufficient evidence (typically 2-3 transcript segments)
- **False positive rate**: 0% in test scenarios — no incorrect identifications
- **Graceful uncertainty**: System never forces a wrong answer; reports "uncertain" when margin is too small

## Confidence Evolution

The EMA smoothing (α=0.3) produces realistic confidence curves:

```
Event:     join    join    join    transcript    transcript    transcript
           (IV)    (IV)    (CAND)  (IV asks)     (CAND answers) (IV asks)
Conf:     50%     50%     52%     55%           62%            68%
Status:   uncert  uncert  uncert  uncert        uncertain      identified
```

This matches production expectations — Sherlock shouldn't lock onto a candidate in the first 30 seconds.

## Limitations

1. **No real meeting data**: Scenarios are synthetic. Production accuracy may differ with real-world noise, accents, and meeting platform quirks.

2. **Rule-based transcript analysis**: Regex patterns for Q&A detection are brittle. The optional LLM boost helps but adds latency (~200ms) and cost.

3. **Static signal weights**: Weights are hand-tuned, not learned from data. A production system should learn optimal weights from labeled interviews.

4. **No voice/face biometrics**: The strongest identity signals (voice fingerprint, face match) are not implemented. These would significantly boost accuracy for generic-name scenarios.

5. **Single candidate assumption**: System assumes exactly one candidate per meeting. Panel interviews with multiple candidates are not handled.

6. **No cross-meeting learning**: Each meeting starts fresh. A production system could use prior interview history for the same candidate.

7. **Speaking duration approximation**: In the prototype, speaking duration comes from simulated events. Real VAD-based duration from separate audio streams would be more accurate.

8. **Platform-specific quirks**: Different meeting platforms have different join patterns, name formats, and observer behaviors not captured in scenarios.

## What Would Improve Accuracy

| Improvement | Expected Impact | Effort |
|-------------|----------------|--------|
| Voice biometrics (pre-interview sample) | +15-20% confidence in ambiguous cases | Medium |
| Face recognition (LinkedIn/calendar photo) | +10-15% for webcam-on scenarios | Medium |
| Learned fusion weights (100+ labeled interviews) | +5-10% overall, better calibration | High |
| Real-time LLM transcript analysis | +5-10% in missing-metadata cases | Low |
| Active learning from human corrections | Continuous improvement | Medium |

## Explainability Quality

Every identification result includes:
- Overall explanation string ("Identified X as candidate with Y% confidence")
- Per-signal evidence breakdown (top 4 contributing signals)
- Participant rankings with per-signal scores
- Status indicator (identified / uncertain / no_participants)

Example output:
```
Identified 'Priya Sharma' as candidate with 75% confidence.
Next closest: 'Alex Chen' (32%).

Evidence:
  • name_match: supports candidate (Display name 'Priya Sharma' closely matches candidate 'Priya Sharma' (100%))
  • transcript_role: supports candidate (Candidate speech patterns detected (2 instances))
  • speaking_pattern: supports candidate (Answer-heavy speech (2 answers vs 0 questions))
  • interviewer_exclusion: supports candidate (Does not match any known interviewer)
```
