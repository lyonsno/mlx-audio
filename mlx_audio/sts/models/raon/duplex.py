import enum
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

AUDIO_OUTPUT_SIL_ID = 151672
AUDIO_OUTPUT_BACKCHANNEL_ID = 151673
AUDIO_START_ID = 151669
AUDIO_OUTPUT_PLACEHOLDER_ID = 151675
AUDIO_INPUT_PLACEHOLDER_ID = 151676
AUDIO_OUTPUT_PAD_ID = 151677
AUDIO_OUTPUT_END_PAD_ID = 151678

_BLOCKED_STRUCTURAL = frozenset(
    {
        151644,  # <|im_start|>
        151645,  # <|im_end|>
        151669,  # <|audio_start|>
        151670,  # <|audio_end|>
        151671,  # <|speaker_embedding_placeholder|>
        AUDIO_OUTPUT_PLACEHOLDER_ID,
        AUDIO_INPUT_PLACEHOLDER_ID,
    }
)


class DuplexPhase(enum.Enum):
    SIL = "SIL"
    SPEECH = "SPEECH"


@dataclass
class DuplexMachineState:
    phase: DuplexPhase
    last_frame_tokens: list[int]

    @property
    def num_input_tokens(self) -> int:
        return len(self.last_frame_tokens)

    @property
    def emitted_audio(self) -> bool:
        return (
            AUDIO_OUTPUT_PLACEHOLDER_ID in self.last_frame_tokens
            or AUDIO_START_ID in self.last_frame_tokens
        )


@dataclass(frozen=True)
class DuplexStateConfig:
    use_duplex_end_pad: bool = False
    use_sil_token: bool = False
    no_audio_in_sil: bool = False
    sequence_mode: Literal["tua", "uta"] | None = None
    duplex_pad_token_id: int = AUDIO_OUTPUT_PAD_ID
    duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD_ID
    duplex_sil_token_id: int = AUDIO_OUTPUT_SIL_ID
    use_backchannel_token: bool = False
    duplex_bc_token_id: int = AUDIO_OUTPUT_BACKCHANNEL_ID

    @property
    def effective_sequence_mode(self) -> Literal["tua", "uta"]:
        return self.sequence_mode or "tua"


class DuplexStateManager:
    def __init__(self, config: DuplexStateConfig) -> None:
        self.config = config

    def initial_state(self, speak_first: bool = False) -> DuplexMachineState:
        return DuplexMachineState(
            phase=DuplexPhase.SIL,
            last_frame_tokens=[
                AUDIO_INPUT_PLACEHOLDER_ID,
                AUDIO_OUTPUT_PLACEHOLDER_ID,
            ],
        )

    def initial_forced_prediction_id(self, speak_first: bool) -> int | None:
        if speak_first:
            if self.config.use_duplex_end_pad:
                return self.config.duplex_end_pad_token_id
            return None
        if self.config.use_sil_token:
            return self.config.duplex_sil_token_id
        return None

    def transition(
        self, state: DuplexMachineState, predicted_id: int
    ) -> tuple[DuplexMachineState, list[int], bool]:
        cfg = self.config
        is_uta = cfg.effective_sequence_mode == "uta"
        is_sil_prediction = predicted_id == cfg.duplex_sil_token_id

        def silence_tokens() -> list[int]:
            return [AUDIO_INPUT_PLACEHOLDER_ID, AUDIO_OUTPUT_PLACEHOLDER_ID]

        def speech_tokens(context_token: int) -> list[int]:
            if is_uta:
                return [
                    AUDIO_INPUT_PLACEHOLDER_ID,
                    context_token,
                    AUDIO_OUTPUT_PLACEHOLDER_ID,
                ]
            return [
                context_token,
                AUDIO_INPUT_PLACEHOLDER_ID,
                AUDIO_OUTPUT_PLACEHOLDER_ID,
            ]

        if state.phase == DuplexPhase.SIL:
            if is_sil_prediction:
                tokens = silence_tokens()
                return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

            if cfg.use_duplex_end_pad and predicted_id == cfg.duplex_end_pad_token_id:
                tokens = speech_tokens(cfg.duplex_end_pad_token_id)
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            if cfg.use_backchannel_token and predicted_id == cfg.duplex_bc_token_id:
                tokens = speech_tokens(cfg.duplex_bc_token_id)
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            tokens = speech_tokens(predicted_id)
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        if is_sil_prediction:
            tokens = silence_tokens()
            return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

        if predicted_id == cfg.duplex_pad_token_id:
            tokens = silence_tokens()
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        tokens = speech_tokens(predicted_id)
        return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

    def apply_logit_mask(
        self, user_logits: mx.array, state: DuplexMachineState, vocab_size: int
    ) -> mx.array:
        cfg = self.config
        token_ids = mx.arange(user_logits.shape[-1])

        if state.phase == DuplexPhase.SIL:
            allowed = token_ids == cfg.duplex_sil_token_id
            if cfg.use_duplex_end_pad:
                allowed = allowed | (token_ids == cfg.duplex_end_pad_token_id)
            if cfg.use_backchannel_token:
                allowed = allowed | (token_ids == cfg.duplex_bc_token_id)
        else:
            context_token = self._extract_context_token(state)
            onset_ids = {cfg.duplex_end_pad_token_id}
            if cfg.use_backchannel_token:
                onset_ids.add(cfg.duplex_bc_token_id)

            if context_token in onset_ids:
                allowed = token_ids < vocab_size
                blocked = (
                    _BLOCKED_STRUCTURAL
                    | {
                        cfg.duplex_sil_token_id,
                        cfg.duplex_pad_token_id,
                    }
                    | onset_ids
                )
                for token_id in blocked:
                    allowed = allowed & (token_ids != token_id)
            elif context_token is not None and context_token not in (
                _BLOCKED_STRUCTURAL
                | onset_ids
                | {cfg.duplex_pad_token_id, cfg.duplex_sil_token_id}
            ):
                allowed = token_ids < vocab_size
                allowed = (
                    allowed
                    | (token_ids == cfg.duplex_pad_token_id)
                    | (token_ids == cfg.duplex_end_pad_token_id)
                    | (token_ids == cfg.duplex_sil_token_id)
                )
                for token_id in _BLOCKED_STRUCTURAL:
                    allowed = allowed & (token_ids != token_id)
            else:
                allowed = (
                    (token_ids == cfg.duplex_pad_token_id)
                    | (token_ids == cfg.duplex_end_pad_token_id)
                    | (token_ids == cfg.duplex_sil_token_id)
                )

        mask = mx.where(allowed, 0.0, float("-inf")).astype(user_logits.dtype)
        return user_logits + mask

    def _extract_context_token(self, state: DuplexMachineState) -> int | None:
        if len(state.last_frame_tokens) != 3:
            return None
        if self.config.effective_sequence_mode == "uta":
            return state.last_frame_tokens[1]
        return state.last_frame_tokens[0]
