# [PACKAGE DOC] Model definitions for the torchtitan backend that torchtitan itself does not ship.
# Each subpackage exposes a model registry function (taking a flavor name and attention backend and
# returning a torchtitan model specification), looked up by name from the model package search path
# before torchtitan's own models. Currently contains the glm4_moe_lite subpackage. This package
# marker itself defines no names.
