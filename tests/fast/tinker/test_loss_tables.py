# [MODULE DOC] Consistency tests between the tinker core's admission tables and the trainer: the
# set of loss function names with admission entries must equal the trainer's registered tinker
# loss functions, and the wire input keys' mapped values must cover exactly the runtime's
# datum-to-batch key names.
"""The admission tables in core must agree with the trainer's loss registry."""

from miles.backends.training_utils.loss.hub.tinker_losses import TINKER_LOSS_FUNCTIONS
from miles.tinker.core.types import LOSS_FN_INPUTS, LOSS_INPUT_KEYS
from miles.tinker.runtime import DATUM_TO_BATCH_KEYS


def test_every_registered_loss_has_an_admission_entry():
    # [FUNCTION DOC] The key set of the core's loss function inputs table equals the key set of the
    # trainer's tinker loss function registry.
    assert set(LOSS_FN_INPUTS) == set(TINKER_LOSS_FUNCTIONS)


def test_wire_keys_map_onto_trainer_batch_keys():
    # [FUNCTION DOC] The set of values of the core's wire-to-internal loss input key mapping equals
    # the set of keys of the runtime's datum-to-batch key mapping.
    assert set(LOSS_INPUT_KEYS.values()) == set(DATUM_TO_BATCH_KEYS)
