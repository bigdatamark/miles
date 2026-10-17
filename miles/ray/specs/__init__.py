# [PACKAGE DOC] Declarative worker specs for a Miles run. Each module describes the pools of workers
# one role deploys (rollout executor, inference controller, routers, session servers, inference
# engines, trainer controllers and trainers) as spec objects holding scheduling, ports, environment,
# launch command or worker class, and constructor arguments; plus naming helpers and provider
# factories for those pools. The entrypoint module combines them and filters by deploy component.
