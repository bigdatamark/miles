# [MODULE DOC] Entry points that compute the full list of worker specs of a run from parsed arguments
# (or from an argument vector), restricted to the deploy component the process is responsible for.
from miles.ray.specs import inference, rollout, train
from miles.utils.arguments import parse_args
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    # [FUNCTION DOC] Return all worker specs whose deploy component is selected by the deploy
    # component in args (the "all" component selects everything), preserving the order of the
    # all-specs helper.
    selector = DeployComponent(args.deploy_component)
    return [spec for spec in _compute_all_specs(args) if selector.selects(spec.deploy_component)]


def _compute_all_specs(args) -> list[BaseWorkerSpec]:
    # [FUNCTION DOC] HELPER: Concatenate, in this order: the rollout executor spec; the inference
    # controller spec; one router spec per model; registration reporter specs (only for an
    # inference-only deployment); the session server spec; inference engine specs; trainer controller
    # specs; trainer specs. Unfiltered.
    return [
        rollout.spec_rollout_executor(args),
        inference.spec_inference_controller(args),
        *inference.specs_router(args),
        *inference.specs_inference_registration_reporter(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        *train.specs_trainer_controller(args),
        *train.specs_trainer(args),
    ]


def compute_specs_from_argv(argv: list[str]) -> list[BaseWorkerSpec]:
    # [FUNCTION DOC] Parse the given argument vector with the global argument parser (temporarily
    # overriding the process argument vector, restored afterwards) and compute the specs from the
    # resulting args. Used by external tooling that only has command-line text.
    with override_argv(argv):
        return compute_specs(parse_args())
