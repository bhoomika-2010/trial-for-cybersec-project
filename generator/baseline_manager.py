class BaselineManager:

    def __init__(
        self,
        user_baselines,
        department_baselines
    ):
        self.user_baselines = user_baselines
        self.department_baselines = department_baselines

    def get_baseline(self, entity_id, department):

        # Personal baseline exists
        if entity_id in self.user_baselines:

            return (
                self.user_baselines[entity_id],
                "user"
            )

        # Cold start:
        # No personal baseline, so use
        # the department baseline.
        if department in self.department_baselines:

            return (
                self.department_baselines[department],
                "department"
            )

        # No baseline available
        return None, None