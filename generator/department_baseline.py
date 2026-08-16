from collections import defaultdict
from statistics import mean, stdev


class DepartmentBaselineGenerator:

    def __init__(self, events, entities, resources):
        self.events = events
        self.entities = entities
        self.resources = resources

    def generate_department_baselines(self):

        department_baselines = {}

        # ==================================================
        # Create entity_id -> entity lookup
        # ==================================================

        entity_lookup = {
            entity["entity_id"]: entity
            for entity in self.entities
        }

        # ==================================================
        # Group normal events by department
        # ==================================================

        department_events = defaultdict(list)

        for event in self.events:

            entity_id = event["entity_id"]

            entity = entity_lookup.get(entity_id)

            if entity is None:
                continue

            department = entity["department"]

            department_events[department].append(event)

        # ==================================================
        # Generate baseline for each department
        # ==================================================

        for department, events in department_events.items():

            if not events:
                continue

            # ==================================================
            # ROLES
            # ==================================================

            roles = set()

            for event in events:

                entity = entity_lookup[event["entity_id"]]

                roles.add(entity["role"])

            roles = sorted(roles)

            # ==================================================
            # NORMAL RESOURCES
            # ==================================================

            # These come directly from Office.resources.
            # This is the authoritative list of resources
            # belonging to this department.

            normal_resources = self.resources[
                department
            ]

            # ==================================================
            # WORKING HOURS
            # ==================================================

            working_start_hours = []
            working_end_hours = []

            # We only need each user's working hours once,
            # not once for every event.

            department_entity_ids = set(
                event["entity_id"]
                for event in events
            )

            for entity_id in department_entity_ids:

                entity = entity_lookup[entity_id]

                start_hour = int(
                    entity["working_start_time"]
                    .split(":")[0]
                )

                end_hour = int(
                    entity["working_end_time"]
                    .split(":")[0]
                )

                working_start_hours.append(
                    start_hour
                )

                working_end_hours.append(
                    end_hour
                )

            avg_working_start_hour = mean(
                working_start_hours
            )

            avg_working_end_hour = mean(
                working_end_hours
            )

            # ==================================================
            # EVENTS PER DAY
            # ==================================================

            # Count events for every user on every day.

            user_day_counts = defaultdict(int)

            for event in events:

                entity_id = event["entity_id"]
                event_date = event["timestamp"].date()

                key = (
                    entity_id,
                    event_date
                )

                user_day_counts[key] += 1

            # Average the daily activity across
            # all users/days in this department.

            daily_event_counts = list(
                user_day_counts.values()
            )

            avg_events_per_day = mean(
                daily_event_counts
            )

            # ==================================================
            # EVENT GAPS
            # ==================================================

            # Group events by user first.
            # We must NEVER calculate a gap between
            # two different users.

            user_events = defaultdict(list)

            for event in events:

                user_events[
                    event["entity_id"]
                ].append(event)

            event_gaps = []

            for entity_id, user_event_list in user_events.items():

                # Group this user's events by date

                events_by_date = defaultdict(list)

                for event in user_event_list:

                    event_date = event["timestamp"].date()

                    events_by_date[event_date].append(
                        event["timestamp"]
                    )

                # Calculate gaps only within the same day

                for timestamps in events_by_date.values():

                    timestamps.sort()

                    for i in range(1, len(timestamps)):

                        gap = (
                            timestamps[i]
                            - timestamps[i - 1]
                        ).total_seconds() / 60

                        event_gaps.append(gap)

            if event_gaps:

                avg_event_gap_minutes = mean(
                    event_gaps
                )

                if len(event_gaps) > 1:

                    std_event_gap_minutes = stdev(
                        event_gaps
                    )

                else:

                    std_event_gap_minutes = 0

            else:

                avg_event_gap_minutes = 0
                std_event_gap_minutes = 0

            # ==================================================
            # AUTHENTICATION FAILURE RATE
            # ==================================================

            total_authentications = len(events)

            failed_authentications = sum(
                1
                for event in events
                if event["auth_result"] == "failure"
            )

            auth_failure_rate = (
                failed_authentications
                / total_authentications
                if total_authentications > 0
                else 0
            )

            # ==================================================
            # DATA DOWNLOADED
            # ==================================================

            downloads = [
                float(event["data_downloaded"])
                for event in events
            ]

            avg_data_downloaded = mean(
                downloads
            )

            if len(downloads) > 1:

                std_data_downloaded = stdev(
                    downloads
                )

            else:

                std_data_downloaded = 0

            # ==================================================
            # STORE DEPARTMENT BASELINE
            # ==================================================

            department_baselines[department] = {

                "department": department,

                "number_of_users":
                    len(department_entity_ids),

                # ------------------------------
                # Roles
                # ------------------------------

                "roles": roles,

                # ------------------------------
                # Resources
                # ------------------------------

                "normal_resources":
                    normal_resources,

                # ------------------------------
                # Working time
                # ------------------------------

                "avg_working_start_hour":
                    avg_working_start_hour,

                "avg_working_end_hour":
                    avg_working_end_hour,

                # ------------------------------
                # Activity
                # ------------------------------

                "avg_events_per_day":
                    avg_events_per_day,

                "avg_event_gap_minutes":
                    avg_event_gap_minutes,

                "std_event_gap_minutes":
                    std_event_gap_minutes,

                # ------------------------------
                # Authentication
                # ------------------------------

                "auth_failure_rate":
                    auth_failure_rate,

                # ------------------------------
                # Data transfer
                # ------------------------------

                "avg_data_downloaded":
                    avg_data_downloaded,

                "std_data_downloaded":
                    std_data_downloaded
            }

        return department_baselines