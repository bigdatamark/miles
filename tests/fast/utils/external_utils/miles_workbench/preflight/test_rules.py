# [MODULE DOC] Test for the preflight permission rule registry: the built in rule sets for the
# chart, the chart service account, the chart RBAC objects and the workflow must equal an exact
# expected mapping of resources to verbs.
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    CHART_RBAC_RULES,
    CHART_RULES,
    CHART_SERVICE_ACCOUNT_RULES,
    WORKFLOW_RULES,
)


class TestBuiltinPermissionRules:
    # [CLASS DOC] Test of the built in permission rules.
    def test_builtin_permission_rule_registry_is_complete(self) -> None:
        # [FUNCTION DOC] The four rule constants taken together equal an expected mapping: chart
        # (config maps, stateful sets, secrets), chart service account (service accounts), chart
        # RBAC (roles and role bindings) and workflow (pods get and list, pod exec create and
        # get, pod log get, stateful sets get list watch), each with their exact verb tuples.
        """The built-in registry contains exactly every externally required permission."""
        expected: dict[str, dict[str, tuple[str, ...]]] = {
            "chart": {
                "configmaps": ("create", "delete", "get", "patch"),
                "statefulsets.apps": ("create", "delete", "get", "patch"),
                "secrets": ("create", "delete", "get", "list", "update"),
            },
            "chart_service_account": {
                "serviceaccounts": ("create", "delete", "get", "patch"),
            },
            "chart_rbac": {
                "roles.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
                "rolebindings.rbac.authorization.k8s.io": ("create", "delete", "get", "patch"),
            },
            "workflow": {
                "pods": ("get", "list"),
                "pods/exec": ("create", "get"),
                "pods/log": ("get",),
                "statefulsets.apps": ("get", "list", "watch"),
            },
        }

        assert {
            "chart": CHART_RULES,
            "chart_service_account": CHART_SERVICE_ACCOUNT_RULES,
            "chart_rbac": CHART_RBAC_RULES,
            "workflow": WORKFLOW_RULES,
        } == expected
