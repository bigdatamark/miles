# [PACKAGE DOC] Example package for multi-policy training: two policies (a solver and a verifier)
# are trained together in one run on GSM8K. Contains the custom generate function and eval
# logging hook that make the verifier check the solver's answer (solver_verifier) and the
# launcher script that configures the two trainers and two inference engine groups
# (run_solver_verifier_gsm8k). The package itself defines no code.
