# [PACKAGE DOC]
# Worker providers: the abstraction through which the rest of the system learns which cells and
# workers exist, where they listen, and how to call them. base defines the interface and cell info
# model; ray, static and kubernetes (core and helm) implement it for each deployment style; utils
# holds helpers shared by implementations.
