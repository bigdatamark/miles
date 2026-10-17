# [PACKAGE DOC]
# HTTP server layer of the Tinker gateway. The app module builds the web application exposing the
# Tinker SDK endpoints; the encoding module translates JSON requests and results; the proto_codec
# module translates the protobuf (and compressed) forms. This layer only converts formats and
# delegates all logic to the core service. The package init itself defines nothing.
