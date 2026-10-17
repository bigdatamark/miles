# [PACKAGE DOC] Loss computation for training: the hub subpackage holds the individual loss
# functions (policy, value, supervised fine-tuning, score centering and others), advantage
# estimators, off-policy corrections, and logit-processing helpers; other modules in this package
# (such as the objective that dispatches and scales the loss) tie them to the training loop. This
# package marker itself defines no names.
