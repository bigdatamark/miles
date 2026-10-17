# [MODULE DOC] Defines the base class of interaction environments used by the multi-turn rollout.
class BaseInteractionEnv:
    # [CLASS DOC] Contract of an interaction environment: reset starts an episode, step consumes the
    # model's latest response text, close releases resources, and format_observation turns an
    # observation mapping into a chat message. Subclasses must implement reset and step.
    """
    Base class that defines the explicit contract for interaction environments.
    """

    def reset(self):
        # [FUNCTION DOC] Abstract: starts an episode. Subclasses return a pair of the initial
        # observation mapping and an info mapping. This base version raises a not-implemented error.
        raise NotImplementedError

    def step(self, response_text: str):
        # [FUNCTION DOC] Abstract: consumes the model's response text for one turn. Subclasses
        # return a triple of observation mapping, done flag and info mapping. This base version raises
        # a not-implemented error.
        raise NotImplementedError

    def close(self):
        # [FUNCTION DOC] Releases resources; the base version does nothing.
        pass

    def format_observation(self, observation: dict) -> dict:
        # [FUNCTION DOC] Converts an observation mapping (none treated as empty) into a user chat
        # message with a list of content parts: first one image part for every image found under the
        # "multi_modal_data" entry (a mapping from modality name to list of images; iteration order
        # of the mapping, regardless of the modality name), then one text part with the "obs_str"
        # entry (empty text if missing). Returns a mapping with role user and that content list.
        # KNOWN ISSUE: any "role" in the observation (the subclass sets tool) is ignored; the message
        # role is always user.
        observation = observation or {}
        content = []
        multimodal = observation.get("multi_modal_data") or {}

        for _, images in multimodal.items():
            for image in images:
                content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": observation.get("obs_str", "")})
        return {"role": "user", "content": content}
