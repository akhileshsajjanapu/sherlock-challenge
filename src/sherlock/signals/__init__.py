"""Signal extractor registry."""

from sherlock.signals.base import SignalExtractor
from sherlock.signals.device_name import DeviceNameSignal
from sherlock.signals.interviewer_exclusion import InterviewerExclusionSignal
from sherlock.signals.join_order import JoinOrderSignal
from sherlock.signals.name_match import NameMatchSignal
from sherlock.signals.observer_silence import ObserverSilenceSignal
from sherlock.signals.screen_share import ScreenShareSignal
from sherlock.signals.speaking_pattern import SpeakingPatternSignal
from sherlock.signals.transcript_role import TranscriptRoleSignal
from sherlock.signals.webcam import WebcamBehaviorSignal


def get_default_signals() -> list[SignalExtractor]:
    return [
        NameMatchSignal(),
        InterviewerExclusionSignal(),
        SpeakingPatternSignal(),
        TranscriptRoleSignal(),
        WebcamBehaviorSignal(),
        DeviceNameSignal(),
        JoinOrderSignal(),
        ScreenShareSignal(),
        ObserverSilenceSignal(),
    ]
