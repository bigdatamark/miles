# [MODULE DOC] Behavior tests of the tinker gateway service (core dispatch logic) over a fake
# backend with two adapter slots, using the shared service fixture (a running service) and the
# harness builders. Covers: idempotent resubmission by sequence id; two-phase model creation and
# slot accounting; user error mapping (no free slots, wrong base model, illegal sequence id,
# unsupported LoRA settings, unknown session); ordering of out-of-order chunks with optimizer
# barriers; failure isolation (a failed training batch or failed step retires only that model and
# frees or quarantines its slot; read-only failures keep queued training); checkpoint save/load
# round trips, overwrite rules, traversal-name rejection, tenant ownership and metadata checks;
# sampler checkpoint publication, naming, versioning and sampling resolution independent of the
# lease; lease expiry reclaiming tenants while checkpoints outlive them; admission validation of
# loss inputs and sample counts; and dispatcher shutdown behavior on unknown failures, dead
# trainers and cancellation. Helper at module level builds an optimizer payload.
"""The service preserves request ordering, tenant isolation, and failure isolation."""

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest
from tests.fast.tinker.harness import (
    ADAM,
    await_settled,
    created_model,
    datum,
    fb_payload,
    make_service,
    model_payload,
)

from miles.tinker.core.future import DONE, FAILED
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.core.utils import resolve_checkpoint_dir, resolve_sampler_checkpoint


def _optim_payload(model_id: str, seq_id: int) -> dict:
    # [FUNCTION DOC] HELPER: Builds an optimizer-step payload mapping with the model id, sequence
    # id and a copy of the default Adam parameters.
    return {"model_id": model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}


async def test_resubmitted_seq_id_reuses_the_future_and_runs_once(service):
    # [FUNCTION DOC] Submitting the same forward/backward payload twice returns the same request
    # id; after it completes (state done), a third submission still returns that id, and the
    # backend ran forward/backward exactly once.
    model_id = await created_model(service)
    payload = fb_payload(model_id, 1, [datum()])

    first = service.submit("tenant", "forward_backward", payload)
    second = service.submit("tenant", "forward_backward", payload)
    assert first == second

    future = await await_settled(service, "tenant", first)
    assert future.state == DONE
    assert service.submit("tenant", "forward_backward", payload) == first, "a resubmit after completion also dedups"
    assert len(service.backend.named("forward_backward")) == 1, "the resubmit must not run the trainer again"


async def test_create_model_is_two_phase(service):
    # [FUNCTION DOC] Creating a model with LoRA rank 8 returns a request id and model id
    # immediately; the future later resolves with the create-model result naming that model, the
    # backend's slot load received rank 8, and the model is registered in the service.
    request_id, model_id = service.create_model("tenant", model_payload(service, lora_config={"rank": 8}))

    future = await await_settled(service, "tenant", request_id)
    assert future.result == {"op": "create_model", "model_id": model_id}
    assert service.backend.named("load_slot")[0]["rank"] == 8
    assert model_id in service.models


async def test_failed_slot_init_returns_the_slot(service):
    # [FUNCTION DOC] If the backend's slot load fails (error mapping), the creation future is
    # failed with category server, the model is not registered, and the set of free slots is
    # unchanged.
    free_before = set(service.free_slots)
    service.backend.fail_next = {"error": "init blew up"}
    request_id, model_id = service.create_model("tenant", model_payload(service))

    future = await await_settled(service, "tenant", request_id)
    assert (future.state, future.error_category) == (FAILED, "server")
    assert model_id not in service.models
    assert service.free_slots == free_before


async def test_no_free_slots_is_a_user_error(service):
    # [FUNCTION DOC] After creating as many models as there are slots, one more creation raises a
    # user input error saying there are no free adapter slots.
    for _ in range(service.config.n_slots):
        await created_model(service)
    with pytest.raises(UserInputError, match="no free adapter slots"):
        service.create_model("tenant", model_payload(service))


async def test_the_wrong_base_model_is_rejected(service):
    # [FUNCTION DOC] Creating a model for a different base model raises a user input error whose
    # message mentions what the service serves.
    with pytest.raises(UserInputError, match="serves"):
        service.create_model("tenant", model_payload(service, base_model="other"))


async def test_out_of_order_chunks_complete_and_the_barrier_waits(service):
    # [FUNCTION DOC] Submits optimizer step 3, then forward/backward 2; after 50 ms the optimizer
    # step has not run (it waits for earlier operations). Submitting forward/backward 1 lets all
    # three finish with state done, and in the backend's call order the optimizer step comes after
    # every forward/backward call.
    model_id = await created_model(service)

    optim = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    late = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    await asyncio.sleep(0.05)
    assert not service.backend.named("optim_step"), "the barrier must wait for every batch op ahead"

    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    for request_id in (early, late, optim):
        assert (await await_settled(service, "tenant", request_id)).state == DONE

    calls = [name for name, _ in service.backend.calls if name != "load_slot"]
    assert calls.index("optim_step") > max(i for i, name in enumerate(calls) if name == "forward_backward")


async def test_admission_failure_terminates_the_training_model_queue(service):
    # [FUNCTION DOC] A forward/backward request with a datum longer than the configured maximum
    # tokens per datum fails; afterwards submitting further training to that model raises a user
    # input error telling to create a new model.
    model_id = await created_model(service)

    oversized = service.submit(
        "tenant", "forward_backward", fb_payload(model_id, 1, [datum(service.config.max_tokens_per_datum + 1)])
    )
    assert (await await_settled(service, "tenant", oversized)).state == FAILED

    with pytest.raises(UserInputError, match="create a new model"):
        service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))


async def test_forward_backward_outputs_align_to_datums(service):
    # [FUNCTION DOC] A request with datums of 2 and 5 target tokens yields outputs whose logprob
    # lists have lengths 2 and 5, in datum order.
    model_id = await created_model(service)
    request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(2), datum(5)]))

    future = await await_settled(service, "tenant", request_id)
    assert [len(output["logprobs"]) for output in future.result["outputs"]] == [2, 5]


async def test_optim_step_returns_the_slot_grad_norm(service):
    # [FUNCTION DOC] After a forward/backward and an optimizer step, the step result's metrics
    # equal the fake backend's gradient norm for the model's slot (0.5 plus slot number).
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 2))

    future = await await_settled(service, "tenant", request_id)
    assert future.result["metrics"] == {"grad_norm": 0.5 + slot}


async def test_save_then_load_roundtrip_paths(service):
    # [FUNCTION DOC] Saves state named ckpt (no overwrite): the result path is the tinker-scheme
    # path model/weights/ckpt. Loading that path without the optimizer finishes done and the
    # backend's last slot load had the optimizer flag false and the resolved on-disk checkpoint
    # path under the checkpoint root.
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt", "overwrite": False}
    )
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    assert saved_path == f"tinker://{model_id}/weights/ckpt"

    load = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": saved_path, "optimizer": False}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    weights_only = service.backend.named("load_slot")[-1]
    assert weights_only["load_optimizer"] is False
    checkpoint = Path(service.config.checkpoint_root) / model_id / "weights" / "ckpt"
    assert weights_only["ckpt_path"] == str(checkpoint.resolve())


async def test_sampler_save_publishes_successive_versions(service):
    # [FUNCTION DOC] Two sampler-weight saves with sequence ids 1 and 2 return paths ending in
    # sampler_weights/1 and /2; the backend's two slot exports went to the matching on-disk
    # sampler-weights version directories.
    model_id = await created_model(service)
    for seq_id in (1, 2):
        request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id})
        future = await await_settled(service, "tenant", request_id)
        assert future.result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
    assert [export["path"] for export in service.backend.named("export_slot")] == [
        resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", str(version))
        for version in (1, 2)
    ]


async def test_sampling_resolves_against_the_saved_version(service):
    # [FUNCTION DOC] After a sampler save, submitting a sample request for 2 samples with that
    # path returns a request id and two sequence ids; the future holds two sequences and the
    # backend's sample call used the adapter name model-id at version 1.
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    sampler_path = (await await_settled(service, "tenant", save)).result["path"]

    request_id, sequence_ids = service.submit_sample(
        "tenant",
        {
            "model_path": sampler_path,
            "num_samples": 2,
            "prompt_tokens": [1, 2],
            "sampling_params": {"max_tokens": 4},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    assert len(sequence_ids) == 2
    future = await await_settled(service, "tenant", request_id)
    assert len(future.result["sequences"]) == 2
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@1"


async def test_sampler_requests_carry_the_published_checkpoint_path(service):
    # [FUNCTION DOC] After a sampler save, the export went to the on-disk version-1 directory, and
    # the later sample call's adapter path is that same directory.
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", request_id)).result["path"]

    disk_dir = resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "sampler_weights", "1")
    assert service.backend.named("export_slot")[0]["path"] == disk_dir

    sample_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": path,
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(service, "tenant", sample_id)
    assert service.backend.named("sample")[0]["lora_path"] == disk_dir


async def test_sample_failure_preserves_the_snapshot_and_training_model_queue(service):
    # [FUNCTION DOC] After a sampler save, makes the backend's next sample fail with engine down;
    # the sample future fails with that error text, the saved sampler checkpoint still resolves
    # to the same adapter name, and a later forward/backward on the model still completes.
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    future = await await_settled(service, "tenant", request_id)
    assert future.result["path"] == f"tinker://{model_id}/sampler_weights/1"

    service.backend.fail_on["sample"] = {"error": "engine down"}
    sample_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": future.result["path"],
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    future = await await_settled(service, "tenant", sample_id)
    assert future.state == FAILED
    assert future.error == "engine down"
    assert (
        resolve_sampler_checkpoint(
            service.config.checkpoint_root,
            "tenant",
            f"tinker://{model_id}/sampler_weights/1",
            service.config.base_model,
        )[0]
        == f"{model_id}@1"
    )
    fb = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    assert (await await_settled(service, "tenant", fb)).state == DONE


async def test_lease_expiry_reclaims_the_tenant(service):
    # [FUNCTION DOC] Creates a session and model, completes an optimizer step, then queues a
    # gapped (sequence 3) forward/backward that stays queued. Ages the session heartbeat beyond
    # the lease timeout and runs the expiry sweep: the model is removed, its slot returns to the
    # free pool, the backend unloaded exactly that slot, and the stale queued future is failed.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    queued = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    await await_settled(service, "tenant", queued)
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [datum()]))  # gapped: stays queued

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()

    assert model_id not in service.models
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve_future("tenant", stale).state == FAILED


async def test_a_fresh_heartbeat_keeps_the_model(service):
    # [FUNCTION DOC] After a heartbeat for the session, the expiry sweep keeps the model.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)

    service.heartbeat("tenant", session_id)
    await service._expire_sessions()

    assert model_id in service.models


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", ".hidden"], ids=["parent", "slash", "dotdot", "hidden"])
async def test_traversal_checkpoint_names_are_rejected(service, name):
    # [FUNCTION DOC] Parametrized over names with parent references, slashes, dot-dot and a
    # hidden name: saving state under such a name fails with category user and an error about an
    # invalid checkpoint path segment, and the backend's save is never called.
    model_id = await created_model(service)
    request_id = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": name, "overwrite": False}
    )
    future = await await_settled(service, "tenant", request_id)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "invalid checkpoint path segment" in future.error
    assert not service.backend.named("save_slot"), "nothing may touch the disk for a rejected name"


async def test_traversal_sampler_paths_are_rejected(service):
    # [FUNCTION DOC] A sample request whose model path climbs out of the weights directory with
    # parent references raises a user input error.
    model_id = await created_model(service)
    with pytest.raises(UserInputError):
        service.submit_sample(
            "tenant",
            {
                "model_path": f"tinker://{model_id}/weights/../../etc",
                "num_samples": 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_expired_result_on_resubmit_fails_instead_of_410(service, monkeypatch):
    # [FUNCTION DOC] Completes a forward/backward, patches the finished-result lifetime negative
    # so the result ages out (retrieval gives none), and resubmits the same command: it gets a
    # different request id. With the lifetime restored, the new future is failed with category
    # user and an expired message, and the backend's forward/backward ran only once (an executed
    # command never re-runs).
    from miles.tinker.core import future as future_module

    model_id = await created_model(service)
    first = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    await await_settled(service, "tenant", first)
    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", -1.0)
    assert service.retrieve_future("tenant", first) is None, "the result must have aged out"

    resubmitted = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert resubmitted != first
    monkeypatch.setattr(future_module, "_FINISHED_TTL_S", 3600.0)  # only the first result aged out
    future = service.retrieve_future("tenant", resubmitted)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "expired" in future.error
    assert len(service.backend.named("forward_backward")) == 1, "an executed command must never re-run"


async def test_save_state_refuses_to_overwrite_unless_asked(service):
    # [FUNCTION DOC] Saving state name ckpt succeeds; saving the same name again without
    # overwrite fails with category user and a message mentioning overwrite; saving with
    # overwrite set succeeds.
    model_id = await created_model(service)
    payload = {"model_id": model_id, "name": "ckpt", "overwrite": False}
    first = service.submit("tenant", "save_state", payload | {"seq_id": 1})
    assert (await await_settled(service, "tenant", first)).state == DONE

    clobber = service.submit("tenant", "save_state", payload | {"seq_id": 2})
    future = await await_settled(service, "tenant", clobber)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "overwrite" in future.error

    replace = service.submit("tenant", "save_state", payload | {"seq_id": 3, "overwrite": True})
    assert (await await_settled(service, "tenant", replace)).state == DONE


async def test_checkpoints_outlive_the_lease(service):
    # [FUNCTION DOC] Saves state, expires the session lease (heartbeat set far in the past) so the
    # model is reclaimed, then creates a new session and model and loads the saved path with the
    # optimizer: the load finishes done.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "kept", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]

    service.sessions[session_id].last_heartbeat = -1e9
    await service._expire_sessions()
    assert model_id not in service.models, "the lease sweep must reclaim the model"

    service.create_session("tenant")
    fresh_id = await created_model(service)
    load = service.submit("tenant", "load_state", {"model_id": fresh_id, "seq_id": 1, "path": path, "optimizer": True})
    assert (await await_settled(service, "tenant", load)).state == DONE


async def test_a_foreign_tenants_checkpoint_does_not_load(service):
    # [FUNCTION DOC] Tenant "tenant" saves state; tenant "thief" with its own model tries to load
    # that path: the future fails with category user and an error about the checkpoint not
    # belonging to the tenant.
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "mine", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]

    thief_model = await created_model(service, tenant="thief")
    load = service.submit(
        "thief", "load_state", {"model_id": thief_model, "seq_id": 1, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "thief", load)
    assert (future.state, future.error_category) == (FAILED, "user")
    assert "belong" in future.error


async def test_a_failed_batch_terminates_the_model_queue(service):
    # [FUNCTION DOC] With two queued forward/backward requests and the backend failing the next
    # call, both futures fail (the failed accumulation terminates training), the model is removed
    # and its slot unloaded.
    model_id = await created_model(service)
    first = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum(), datum()]))
    second = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    service.backend.fail_next = {"error": "batch input rejected"}
    assert (await await_settled(service, "tenant", first)).state == FAILED
    assert (
        await await_settled(service, "tenant", second)
    ).state == FAILED, "the failed accumulation terminates model training"
    assert model_id not in service.models
    assert service.backend.named("unload_slot")


async def test_merged_optim_settles_each_slot_on_its_own(service):
    # [FUNCTION DOC] Two models' optimizer steps merge in one backend call; the second model's
    # slot is configured to return an error. The first future is done with a gradient norm
    # metric; the second fails with category server and an error containing boom.
    model_a = await created_model(service)
    model_b = await created_model(service)
    slot_b = service.models[model_b].slot
    service.backend.optim_outcomes[slot_b] = {"error": "boom"}

    ok = service.submit("tenant", "optim_step", _optim_payload(model_a, 1))
    bad = service.submit("tenant", "optim_step", _optim_payload(model_b, 1))
    resolved = await await_settled(service, "tenant", ok)
    failed = await await_settled(service, "tenant", bad)
    assert resolved.state == DONE and "grad_norm" in resolved.result["metrics"]
    assert (failed.state, failed.error_category) == (FAILED, "server") and "boom" in failed.error


async def test_a_nonfinite_step_reports_the_skip(service):
    # [FUNCTION DOC] If the backend reports a skipped-nonfinite outcome for the slot, the optimizer
    # step future is done and its metrics are exactly that skipped marker.
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.optim_outcomes[slot] = {"skipped_nonfinite": 1.0}
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", request_id)
    assert future.state == DONE
    assert future.result["metrics"] == {"skipped_nonfinite": 1.0}


async def test_num_samples_is_capped(service):
    # [FUNCTION DOC] After a sampler save, a sample request with one more than the configured
    # maximum samples per request raises a user input error.
    model_id = await created_model(service)
    version = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", version)).result["path"]
    with pytest.raises(UserInputError):
        service.submit_sample(
            "tenant",
            {
                "model_path": path,
                "num_samples": service.config.max_samples_per_request + 1,
                "prompt_tokens": [1],
                "sampling_params": {"max_tokens": 2},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
        )


async def test_malformed_loss_inputs_are_rejected_at_admission(service):
    # [FUNCTION DOC] Four malformed cases (unknown loss name, missing required input, wrong input
    # length, an input that the loss never reads such as weights with an importance sampling
    # loss). For each, a fresh model is created and the request submitted; each future fails with
    # category user. Finally the backend's forward/backward was never called.
    cases = {
        "unknown loss_fn": {"loss_fn": "made_up", "datums": [datum()]},
        "missing input": {"loss_fn": "importance_sampling", "datums": [datum()]},
        "wrong length": {
            "loss_fn": "importance_sampling",
            "datums": [datum() | {"sampling_logprobs": [0.0], "advantages": [1.0, 2.0, 3.0, 4.0]}],
        },
        # datum() carries weights, which importance_sampling never reads
        "unread input": {
            "loss_fn": "importance_sampling",
            "datums": [datum() | {"sampling_logprobs": [0.0] * 3, "advantages": [1.0] * 3}],
        },
    }
    for payload in cases.values():
        model_id = await created_model(service)
        seq_id = 1
        request_id = service.submit(
            "tenant",
            "forward_backward",
            {"model_id": model_id, "seq_id": seq_id, "loss_fn_config": {}, **payload},
        )
        future = await await_settled(service, "tenant", request_id)
        assert (future.state, future.error_category) == (FAILED, "user")
    assert not service.backend.named("forward_backward"), "rejected datums must never reach the trainer"


async def test_a_failed_window_stops_already_queued_training(service):
    # [FUNCTION DOC] One model completes a forward/backward; then the backend is set to fail the
    # next call and four more commands are queued (forward/backward, step, forward/backward,
    # step). All four fail, the earlier result stays done, no optimizer step reached the backend,
    # the backend saw two forward/backward calls, further training on the model raises the create
    # a new model user error, and a healthy second model's step still completes.
    model_id = await created_model(service)
    healthy = await created_model(service)
    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", early)).state == DONE

    service.backend.fail_next = {"error": "batch rejected"}
    queued = [service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))]
    queued.append(service.submit("tenant", "optim_step", _optim_payload(model_id, 3)))
    queued.append(service.submit("tenant", "forward_backward", fb_payload(model_id, 4, [datum()])))
    queued.append(service.submit("tenant", "optim_step", _optim_payload(model_id, 5)))
    for request_id in queued:
        assert (await await_settled(service, "tenant", request_id)).state == FAILED
    assert service.retrieve_future("tenant", early).state == DONE
    assert not service.backend.named("optim_step")
    assert len(service.backend.named("forward_backward")) == 2
    with pytest.raises(UserInputError, match="create a new model"):
        service.submit("tenant", "optim_step", _optim_payload(model_id, 6))
    step = service.submit("tenant", "optim_step", _optim_payload(healthy, 1))
    assert (await await_settled(service, "tenant", step)).state == DONE


async def test_unsupported_lora_configs_are_rejected(service):
    # [FUNCTION DOC] Creating a model with a LoRA seed, with unembed training, or with MLP
    # training disabled raises a user input error; the supported combination (rank 8, attention
    # and MLP trained, unembed not) is accepted and registered.
    for lora_config in ({"rank": 8, "seed": 7}, {"rank": 8, "train_unembed": True}, {"rank": 8, "train_mlp": False}):
        with pytest.raises(UserInputError):
            service.create_model("tenant", model_payload(service, lora_config=lora_config))
    _, model_id = service.create_model(
        "tenant",
        model_payload(service, lora_config={"rank": 8, "train_attn": True, "train_mlp": True, "train_unembed": False}),
    )
    assert model_id in service.models


@pytest.mark.parametrize("expire_lease", [False, True])
async def test_sampler_paths_resolve_independently_of_the_lease(service, expire_lease):
    # [FUNCTION DOC] Parametrized on whether the lease is expired first. After a sampler save
    # (and optional lease expiry that removes the model and a new session), resolving the sampler
    # path for the owning tenant gives adapter name model-id at version 1 and a path ending in
    # the version directory; another tenant gets a user input or ownership error; and after
    # changing the service's base model to a different one, resolution raises a user input error
    # mentioning base_model. Mutates the service config in place.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]

    if expire_lease:
        service.sessions[session_id].last_heartbeat = -1e9
        await service._expire_sessions()
        assert model_id not in service.models
        service.create_session("tenant")
    lora_name, lora_path = resolve_sampler_checkpoint(
        service.config.checkpoint_root, "tenant", path, service.config.base_model
    )
    assert lora_name == f"{model_id}@1" and lora_path.endswith("/sampler_weights/1")
    with pytest.raises((UserInputError, OwnershipError)):
        resolve_sampler_checkpoint(service.config.checkpoint_root, "thief", path, service.config.base_model)
    service.config.base_model = "other"
    with pytest.raises(UserInputError, match="base_model"):
        resolve_sampler_checkpoint(service.config.checkpoint_root, "tenant", path, service.config.base_model)


async def test_an_unnamed_sampler_save_returns_a_sampling_session(service):
    # [FUNCTION DOC] A sampler save with no sampler path returns a sampling session id in its
    # result, registered on the service for the tenant with the result's path; a sampler save with
    # a name returns no sampling session id.
    model_id = await created_model(service)
    unnamed = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "sampler_path": None}
    )
    result = (await await_settled(service, "tenant", unnamed)).result
    session = service.sampling_sessions[result["sampling_session_id"]]
    assert (session.tenant, session.model_path) == ("tenant", result["path"])

    named = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "sampler_path": "ckpt"}
    )
    assert "sampling_session_id" not in (await await_settled(service, "tenant", named)).result


async def test_weights_info_reads_the_checkpoint_not_the_lease(service):
    # [FUNCTION DOC] After saving state and deleting the model from the service, the weights info
    # for the path returns base model, is-LoRA true, rank 8, attention and MLP trained and
    # unembed not trained (read from the checkpoint metadata); another tenant gets an ownership
    # error.
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", save)).result["path"]
    del service.models[model_id]

    info = service.weights_info("tenant", path)
    assert info == {
        "base_model": "base",
        "is_lora": True,
        "lora_rank": 8,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }
    with pytest.raises(OwnershipError):
        service.weights_info("thief", path)


async def test_a_failed_optim_step_retires_the_model(service):
    # [FUNCTION DOC] With the slot's optimizer outcome set to an error, the step future fails with
    # that message; the model is removed, its slot is free and was unloaded, and a later step on
    # the model raises a user input error telling to restore from a checkpoint.
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.optim_outcomes[slot] = {"error": "slot update rejected"}

    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    future = await await_settled(service, "tenant", step)
    assert future.state == FAILED and "slot update rejected" in future.error
    assert model_id not in service.models, "a half-applied step may have diverged the slot across ranks"
    assert slot in service.free_slots
    assert service.backend.named("unload_slot") == [{"slot": slot}]

    with pytest.raises(UserInputError, match="restore from a checkpoint"):
        service.submit("tenant", "optim_step", _optim_payload(model_id, 2))


async def test_an_out_of_order_validation_failure_blocks_later_training(service):
    # [FUNCTION DOC] Submits step 3, then an invalid forward/backward 2 (unknown loss, no datums),
    # then valid forward/backward 1. Sequence 1 completes, 2 and 3 fail, and the backend ran only
    # one forward/backward and no optimizer step.
    model_id = await created_model(service)
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    rejected = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [], loss_fn="unknown"))
    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", early)).state == DONE
    assert (await await_settled(service, "tenant", rejected)).state == FAILED
    assert (await await_settled(service, "tenant", step)).state == FAILED
    assert len(service.backend.named("forward_backward")) == 1
    assert not service.backend.named("optim_step")


async def test_a_backend_level_optim_failure_retires_every_model_in_the_barrier(service):
    # [FUNCTION DOC] With the whole optimizer call set to fail, two models' merged steps both fail
    # with a message about restoring from a checkpoint; both models are removed with two slot
    # unloads; a new model can still be created afterwards (the dispatch loop survived).
    model_a = await created_model(service)
    model_b = await created_model(service)
    service.backend.fail_on["optim_step"] = {"error": "slot updates rejected"}

    first = service.submit("tenant", "optim_step", _optim_payload(model_a, 1))
    second = service.submit("tenant", "optim_step", _optim_payload(model_b, 1))
    for request_id in (first, second):
        future = await await_settled(service, "tenant", request_id)
        assert future.state == FAILED and "restore from a checkpoint" in future.error
    assert model_a not in service.models and model_b not in service.models
    assert len(service.backend.named("unload_slot")) == 2

    fresh = await created_model(service)  # the dispatch loop survived the failure
    assert fresh in service.models


async def test_a_failed_unload_keeps_the_slot_out_of_the_free_pool(service):
    # [FUNCTION DOC] With the backend's unload set to fail, expiring the session removes the model
    # but the slot stays out of the free pool (it is dirty); a second sweep still runs without
    # error.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    service.backend.fail_on["unload_slot"] = {"error": "engine gone"}

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()

    assert model_id not in service.models
    assert slot not in service.free_slots, "a slot whose unload failed is dirty and must not be reused"
    await service._expire_sessions()  # the sweep itself survived


async def test_a_named_sampler_save_uses_the_name_and_rejects_reuse(service):
    # [FUNCTION DOC] A sampler save with name v1 returns path model/sampler_weights/v1 without a
    # sampling session id; sampling from it uses adapter name model-id at v1; saving again with
    # the same name fails with a message that it already exists (saved versions are immutable).
    model_id = await created_model(service)
    save = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "sampler_path": "v1"}
    )
    result = (await await_settled(service, "tenant", save)).result
    assert result["path"] == f"tinker://{model_id}/sampler_weights/v1"
    assert "sampling_session_id" not in result

    request_id, _ = service.submit_sample(
        "tenant",
        {
            "model_path": result["path"],
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(service, "tenant", request_id)
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@v1"

    reuse = service.submit(
        "tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 2, "sampler_path": "v1"}
    )
    future = await await_settled(service, "tenant", reuse)
    assert future.state == FAILED and "already exist" in future.error, "saved versions are immutable"


async def test_checkpoint_meta_stores_a_digest_not_the_credential(service):
    # [FUNCTION DOC] After saving state, reads the checkpoint's metadata file as JSON and asserts
    # it contains no entry for the tenant name, that the stored tenant digest differs from the
    # tenant name (the bearer credential is not persisted), and that weights info reports
    # attention and MLP trained and unembed not.
    import json
    import os

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    await await_settled(service, "tenant", saved)
    meta_path = os.path.join(
        resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck"), "META.json"
    )
    meta = json.loads(open(meta_path).read())
    assert "tenant" not in meta and meta["tenant_digest"] != "tenant", "the bearer credential must not be persisted"
    info = service.weights_info("tenant", f"tinker://{model_id}/weights/ck")
    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (True, True, False)


async def test_a_checkpoint_saved_under_other_settings_does_not_load(service):
    # [FUNCTION DOC] Saves state, rewrites its metadata file with the LoRA alpha increased by
    # one, then loads the checkpoint: the future fails with category user and an error mentioning
    # lora_alpha, and the backend's slot load was not called again (only the initial load exists).
    import json
    import os

    model_id = await created_model(service)
    saved = service.submit(
        "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
    )
    path = (await await_settled(service, "tenant", saved)).result["path"]
    meta_path = os.path.join(
        resolve_checkpoint_dir(service.config.checkpoint_root, model_id, "weights", "ck"), "META.json"
    )
    meta = json.loads(open(meta_path).read())
    meta["lora_alpha"] = meta["lora_alpha"] + 1  # the same tensors would be scaled differently
    open(meta_path, "w").write(json.dumps(meta))

    loaded = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    future = await await_settled(service, "tenant", loaded)
    assert (future.state, future.error_category) == (FAILED, "user") and "lora_alpha" in future.error
    assert not service.backend.named("load_slot")[1:], "nothing may touch the slot on a mismatch"


async def test_a_recycled_slot_belongs_to_a_fresh_model_queue(service):
    # [FUNCTION DOC] A model's batch fails (retiring it); then the lease expires so the slot is
    # freed and the session removed. A new model takes the same slot; its forward/backward and
    # optimizer step both complete (state done), showing the recycled slot has a fresh queue.
    session_id = service.create_session("tenant")
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    service.backend.fail_next = {"error": "batch input rejected"}
    failed = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
    assert (await await_settled(service, "tenant", failed)).state == FAILED

    service.sessions[session_id].last_heartbeat -= service.config.lease_timeout_s + 1
    await service._expire_sessions()
    assert slot in service.free_slots

    assert session_id not in service.sessions, "the sweep takes the expired session with it"
    fresh = await created_model(service)
    assert service.models[fresh].slot == slot
    step_after_fb = service.submit("tenant", "forward_backward", fb_payload(fresh, 1, [datum()]))
    assert (await await_settled(service, "tenant", step_after_fb)).state == DONE
    step = service.submit("tenant", "optim_step", _optim_payload(fresh, 2))
    future = await await_settled(service, "tenant", step)
    assert future.state == DONE, "a fresh model may use the successfully recycled slot"


@pytest.mark.parametrize("source", ["handler", "worker", "create", "sweep", "export", "load"])
async def test_an_unknown_failure_stops_the_dispatcher(tmp_path, source):
    # [FUNCTION DOC] Parametrized over the failure source (handler, worker, create, sweep, export,
    # load). Builds its own service (not the shared fixture) in the temporary directory and runs
    # it as a task; creates a model; then injects an unexpected error: an IO error for export and
    # load, a runtime error otherwise, delivered via the backend failure mapping (export, load,
    # create, optimizer step) or by replacing the barrier handler with a raising coroutine, or via
    # a failed unload during a forced lease sweep run as an observed background task. Asserts the
    # dispatcher task finishes (2 second timeout) by raising that same error type and message.
    # In cleanup cancels the run task if still running.
    from tests.fast.tinker.harness import make_service

    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    error = (
        OSError("checkpoint IO failed") if source in ("export", "load") else RuntimeError("fatal execution failure")
    )
    try:
        model_id = await created_model(gateway)
        if source == "export":
            gateway.backend.fail_on["export_slot"] = error
            gateway.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
        elif source == "load":
            saved = gateway.submit(
                "tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ck", "overwrite": False}
            )
            path = (await await_settled(gateway, "tenant", saved)).result["path"]
            gateway.backend.fail_on["load_slot"] = error
            gateway.submit(
                "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
            )
        elif source == "create":
            gateway.backend.fail_on["load_slot"] = RuntimeError("fatal execution failure")
            gateway.create_model("tenant", model_payload(gateway))
        elif source == "sweep":
            gateway.backend.fail_on["unload_slot"] = RuntimeError("fatal execution failure")
            for session in gateway.sessions.values():
                session.last_heartbeat -= gateway.config.lease_timeout_s + 1
            task = asyncio.create_task(gateway._expire_sessions())
            task.add_done_callback(gateway._observe_background_task)
        else:
            if source == "handler":

                async def broken_handler(unit):
                    # [FUNCTION DOC] HELPER: Async stand-in for the barrier handler; ignores the
                    # unit and always raises a runtime error saying fatal execution failure.
                    raise RuntimeError("fatal execution failure")

                gateway._run_barrier = broken_handler
            else:
                gateway.backend.fail_on["optim_step"] = RuntimeError("fatal execution failure")
            gateway.submit("tenant", "optim_step", _optim_payload(model_id, 1))
        with pytest.raises(type(error), match=str(error)):
            await asyncio.wait_for(run_task, timeout=2)
    finally:
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


async def test_a_dead_trainer_escapes_the_dispatch_loop(tmp_path, monkeypatch):
    # [FUNCTION DOC] Runs its own service as a task and creates a model; replaces the backend's
    # trainer-dead check with one returning true (reverted by monkeypatch) and submits a
    # forward/backward. The dispatcher task must end within 2 seconds by raising a runtime error
    # saying the trainer workers died. Cleanup cancels the task if still running.
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    try:
        model_id = await created_model(gateway)

        async def _dead() -> bool:
            # [FUNCTION DOC] HELPER: Async fake health check that always reports the trainer dead.
            return True

        monkeypatch.setattr(gateway.backend, "trainer_dead", _dead)
        gateway.submit("tenant", "forward_backward", fb_payload(model_id, 1, [datum()]))
        with pytest.raises(RuntimeError, match="trainer workers died"):
            await asyncio.wait_for(run_task, timeout=2)
    finally:
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


async def test_a_retried_create_model_returns_the_first_allocation(service):
    # [FUNCTION DOC] Creating a model twice with the same payload returns an equal result (request
    # id and model id) and only one slot load reaches the backend.
    """A network retry must not occupy a second slot."""
    payload = model_payload(service)
    first = service.create_model("tenant", payload)
    await await_settled(service, "tenant", first[0])
    assert service.create_model("tenant", payload) == first
    assert len(service.backend.named("load_slot")) == 1


async def test_an_unknown_session_is_rejected(service):
    # [FUNCTION DOC] Creating a model for a nonexistent session id raises a user input error
    # saying unknown session.
    with pytest.raises(UserInputError, match="unknown session"):
        service.create_model("tenant", model_payload(service, session_id="session-gone"))


async def test_a_retried_sampling_session_returns_the_same_id(service):
    # [FUNCTION DOC] Creating a sampling session twice with the same session id and sampling
    # session sequence id returns equal results.
    payload = {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1}
    assert service.create_sampling_session("tenant", payload) == service.create_sampling_session("tenant", payload)


async def test_a_retried_sample_does_not_sample_again(service):
    # [FUNCTION DOC] Creates a sampling session from a saved sampler path, submits a sample
    # request, and resubmits the identical request: the result (request id and sequence ids) is
    # equal, and the backend's sample ran exactly once.
    model_id = await created_model(service)
    save = service.submit("tenant", "save_weights_for_sampler", {"model_id": model_id, "seq_id": 1})
    path = (await await_settled(service, "tenant", save)).result["path"]
    sampling_session_id = service.create_sampling_session(
        "tenant", {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1, "model_path": path}
    )
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 1,
        "num_samples": 1,
        "prompt_tokens": [1],
        "sampling_params": {"max_tokens": 2},
    }
    first = service.submit_sample("tenant", payload)
    assert service.submit_sample("tenant", payload) == first, "the retry must reuse the request and sequence ids"
    await await_settled(service, "tenant", first[0])
    assert len(service.backend.named("sample")) == 1


async def test_an_illegal_seq_id_is_rejected(service):
    # [FUNCTION DOC] An optimizer step with sequence id 0 raises a user input error mentioning seq_id.
    """seq_id 0 would never enter the ordered model queue; every future would pend."""
    model_id = await created_model(service)
    with pytest.raises(UserInputError, match="seq_id"):
        service.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 0, "adam_params": dict(ADAM)})


async def test_an_expired_lease_sweeps_its_sessions(service):
    # [FUNCTION DOC] After a session's heartbeat is set far in the past and the sweep runs,
    # creating a model on that session raises a user input error saying unknown session.
    session_id = service.create_session("tenant")
    service.sessions[session_id].last_heartbeat = -1e9
    await service._expire_sessions()
    with pytest.raises(UserInputError, match="unknown session"):
        service.create_model("tenant", model_payload(service, session_id=session_id))


async def test_a_sampling_session_for_another_base_model_is_rejected(service):
    # [FUNCTION DOC] Creating a sampling session naming a different base model raises a user input
    # error mentioning what the service serves.
    payload = {"session_id": service.create_session("tenant"), "sampling_session_seq_id": 1, "base_model": "other"}
    with pytest.raises(UserInputError, match="serves"):
        service.create_sampling_session("tenant", payload)


async def test_a_read_only_batch_failure_preserves_queued_training(service):
    # [FUNCTION DOC] With the backend's forward-only set to fail, a forward-only request fails
    # while the queued forward/backward and optimizer step both complete (done).
    model_id = await created_model(service)
    service.backend.fail_on["forward_only"] = {"error": "request rejected"}
    readonly = service.submit("tenant", "forward_only", fb_payload(model_id, 1, [datum()]))
    backward = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [datum()]))
    step = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    assert (await await_settled(service, "tenant", readonly)).state == FAILED
    assert (await await_settled(service, "tenant", backward)).state == DONE
    assert (await await_settled(service, "tenant", step)).state == DONE


@pytest.mark.parametrize("shutdown", ["cancel", "failure"])
async def test_dispatcher_shutdown_stops_model_creation_and_sampling(tmp_path, monkeypatch, shutdown):
    # [FUNCTION DOC] Parametrized over cancelling the dispatcher or making its lease sweeper fail.
    # Builds its own service; replaces the backend's slot load and sample with calls that block
    # forever (recording their tasks) and the lease sweeper with one that raises after an event.
    # Starts the dispatcher, starts a model creation and a sample submission, waits (2 seconds
    # each) for both backend calls to start, then either cancels the dispatcher (expecting a
    # cancellation error) or triggers the sweeper failure (expecting its runtime error). Asserts
    # both blocked backend tasks are done afterwards (none outlives the dispatcher). Cleanup
    # cancels and gathers all tasks.
    gateway = make_service(tmp_path)
    started = asyncio.Queue()
    fail_sweep = asyncio.Event()

    async def blocked_backend_call(*args):
        # [FUNCTION DOC] HELPER: Async fake backend call; records the current task in the queue
        # then blocks forever on an event nobody sets.
        started.put_nowait(asyncio.current_task())
        await asyncio.Event().wait()

    async def _run_lease_sweeper():
        # [FUNCTION DOC] HELPER: Async fake sweeper; waits for the trigger event then raises a
        # runtime error saying the lease sweeper failed.
        await fail_sweep.wait()
        raise RuntimeError("lease sweeper failed")

    monkeypatch.setattr(gateway.backend, "load_slot", blocked_backend_call)
    monkeypatch.setattr(gateway.backend, "sample", blocked_backend_call)
    monkeypatch.setattr(gateway, "_run_lease_sweeper", _run_lease_sweeper)
    run_task = asyncio.create_task(gateway.run())
    gateway.create_model("tenant", model_payload(gateway))
    gateway.submit_sample("tenant", {"num_samples": 1, "prompt_tokens": [1]})
    tasks = [run_task, *gateway._create_tasks, *(task for task, _ in gateway._sample_tasks.values())]
    try:
        backend_tasks = [await asyncio.wait_for(started.get(), timeout=2) for _ in range(2)]
        if shutdown == "cancel":
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run_task, timeout=2)
        else:
            fail_sweep.set()
            with pytest.raises(RuntimeError, match="lease sweeper failed"):
                await asyncio.wait_for(run_task, timeout=2)
        assert all(task.done() for task in backend_tasks), "backend tasks outlived the dispatcher"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
