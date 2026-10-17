# [PACKAGE DOC]
# The Miles router package: a lightweight standalone HTTP load-balancing proxy placed in front of
# the SGLang inference worker engines during rollouts. The config module holds the router's
# validated configuration and its derivation from launch args; the router module holds the web
# app (worker registration, least-active-requests routing, background health checks that
# quarantine failing workers, and a catch-all proxy). The package init itself defines nothing.
