import random
import copy
from datetime import timedelta
from faker import Faker


fake = Faker()


IMPOSSIBLE_TRAVEL_LOCATIONS = {
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8683, 151.2093),
    "Singapore": (1.3521, 103.8198),
}


class AnomalyInjector:
    """
    Inject synthetic attack episodes into an ALREADY SEPARATED dataset.

    The caller gives this injector:
        - the events belonging to one split
        - entities
        - resources
        - start_time of that split
        - end_time of that split

    The injector never performs the train/test split.

    Attack events are NEW events with realistic timestamps. Existing normal
    events are not relabelled into attacks.

    Every attack has a required duration. An anchor is only selected when
    the complete attack can fit inside [start_time, end_time].
    """

    ATTACK_DURATIONS = {
        "brute_force": timedelta(minutes=10),
        "credential_stuffing": timedelta(minutes=15),
        "lateral_movement": timedelta(minutes=20),
        "device_spoofing": timedelta(minutes=5),
        "impossible_travel": timedelta(minutes=30),
        "privilege_escalation": timedelta(minutes=30),
        "low_slow_exfiltration": timedelta(days=15),
    }

    def __init__(
        self,
        events,
        entities,
        resources,
        start_time=None,
        end_time=None,
        entity_bounds=None,
    ):
        self.events = list(events)
        self.entities = entities
        self.resources = resources

        # Per-entity split boundaries are essential.
        # A global TRAIN/TEST boundary is not sufficient because
        # each entity is split independently by event count.
        self.entity_bounds = entity_bounds or {}

        # If boundaries are not supplied, infer them from this dataset.
        timestamps = [
            event["timestamp"]
            for event in self.events
            if event.get("timestamp") is not None
        ]

        if not timestamps:
            raise ValueError("No timestamps found in events.")

        self.start_time = start_time or min(timestamps)
        self.end_time = end_time or max(timestamps)

        if self.start_time >= self.end_time:
            raise ValueError(
                "Dataset start_time must be earlier than end_time."
            )

    # =========================================================
    # GENERAL HELPERS
    # =========================================================

    def _timestamp(self, value):
        """
        Convert timestamps to a consistent datetime-like object.
        """
        return value

    def _get_entity_bounds(self, entity_id):
        """
        Return the chronological [start, end] boundary for one entity
        inside the current TRAIN or TEST split.
        """
        if entity_id in self.entity_bounds:
            return self.entity_bounds[entity_id]

        # Fallback for direct use of the injector outside the dataset builder.
        entity_events = [
            event
            for event in self.events
            if event["entity_id"] == entity_id
        ]

        if not entity_events:
            return None

        timestamps = [
            event["timestamp"]
            for event in entity_events
        ]

        return min(timestamps), max(timestamps)

    def _valid_anchor(self, anchor, duration, entity_id=None):
        """
        An attack beginning at anchor must finish inside the CURRENT
        split for that specific entity.

        This is the critical leakage protection.
        """
        if entity_id is not None:
            bounds = self._get_entity_bounds(entity_id)

            if bounds is None:
                return False

            entity_start, entity_end = bounds

            return (
                entity_start <= anchor
                and anchor + duration <= entity_end
            )

        return (
            self.start_time <= anchor
            and anchor + duration <= self.end_time
        )

    def _choose_anchor(self, duration, entity_id=None):
        """
        Choose an existing event that leaves enough room for the entire
        attack inside that entity's current split.
        """
        if entity_id is None:
            candidates = self.events
        else:
            candidates = [
                event
                for event in self.events
                if event["entity_id"] == entity_id
            ]

        valid = [
            event
            for event in candidates
            if self._valid_anchor(
                event["timestamp"],
                duration,
                entity_id,
            )
        ]

        if not valid:
            return None

        return random.choice(valid)

    def _choose_common_anchor(self, duration, entity_ids):
        """
        Credential stuffing needs ONE short attack window shared by
        multiple users.

        Find the intersection of all selected users' valid time ranges.
        This prevents the same source IP from generating an event for a
        user after that user's TRAIN/TEST boundary.
        """
        bounds = []

        for entity_id in entity_ids:
            entity_bounds = self._get_entity_bounds(entity_id)

            if entity_bounds is None:
                return None

            entity_start, entity_end = entity_bounds

            latest_start = entity_start
            latest_end = entity_end - duration

            if latest_end < latest_start:
                return None

            bounds.append(
                (latest_start, latest_end)
            )

        common_start = max(
            start for start, _ in bounds
        )
        common_end = min(
            end for _, end in bounds
        )

        if common_start > common_end:
            return None

        total_seconds = int(
            (common_end - common_start).total_seconds()
        )

        if total_seconds <= 0:
            return common_start

        return (
            common_start
            + timedelta(
                seconds=random.randint(
                    0,
                    total_seconds,
                )
            )
        )

    def _copy_event(self, base_event, timestamp):
        event = copy.deepcopy(base_event)
        event["timestamp"] = timestamp
        return event

    def get_entity_events(self, entity_id):
        return sorted(
            [
                event
                for event in self.events
                if event["entity_id"] == entity_id
            ],
            key=lambda x: x["timestamp"],
        )

    def get_entity(self, entity_id):
        for entity in self.entities:
            if entity["entity_id"] == entity_id:
                return entity
        return None

    def add_events(self, new_events):
        if not new_events:
            return 0

        self.events.extend(new_events)

        self.events.sort(
            key=lambda x: (
                x["entity_id"],
                x["timestamp"],
            )
        )

        return len(new_events)

    def can_fit_attack(self, attack_type, entity_id=None):
        duration = self.ATTACK_DURATIONS[attack_type]
        return self._choose_anchor(
            duration,
            entity_id,
        ) is not None

    # =========================================================
    # 1. BRUTE FORCE
    # =========================================================
    #
    # Same user
    # same attacker/source IP
    # many authentication failures
    # within a few minutes
    #
    # Example:
    #
    # 10:00 normal
    # 10:00:12 failure
    # 10:01:01 failure
    # 10:01:38 failure
    # ...
    #
    # This is deliberately NOT created by relabelling distant
    # existing events.
    # =========================================================

    def add_brute_force(self, entity_id):
        duration = self.ATTACK_DURATIONS["brute_force"]

        base_event = self._choose_anchor(
            duration,
            entity_id,
        )

        if base_event is None:
            return 0

        number_of_events = random.randint(10, 15)

        attack_start = base_event["timestamp"]

        # Increasing offsets inside the 10-minute attack window.
        offsets = sorted(
            random.randint(5, int(duration.total_seconds() - 5))
            for _ in range(number_of_events)
        )

        attack_events = []

        for offset in offsets:
            timestamp = attack_start + timedelta(
                seconds=offset
            )

            event = self._copy_event(
                base_event,
                timestamp,
            )

            event["auth_result"] = "failure"
            event["resource_accessed"] = None
            event["data_downloaded"] = 0
            event["command_sequence"] = []
            event["label"] = "brute_force"

            attack_events.append(event)

        return self.add_events(attack_events)

    # =========================================================
    # 2. CREDENTIAL STUFFING
    # =========================================================
    #
    # Same source IP -> MANY DISTINCT USERS
    #
    # We choose one global attack window and create one failed
    # authentication for each targeted user inside that window.
    #
    # This is intentionally different from brute force.
    # =========================================================

    def add_credential_stuffing(self, entity_ids):
        if len(entity_ids) < 2:
            return 0

        duration = self.ATTACK_DURATIONS[
            "credential_stuffing"
        ]

        # Make sure users are DISTINCT.
        unique_entity_ids = list(dict.fromkeys(entity_ids))

        # One common attack window must be valid for EVERY targeted
        # user's current TRAIN/TEST timeline.
        attack_start = self._choose_common_anchor(
            duration,
            unique_entity_ids,
        )

        if attack_start is None:
            return 0

        attacker_ip = fake.ipv4_private()

        offsets = sorted(
            random.randint(
                5,
                int(duration.total_seconds() - 5),
            )
            for _ in unique_entity_ids
        )

        attack_events = []

        for entity_id, offset in zip(
            unique_entity_ids,
            offsets,
        ):
            entity_events = self.get_entity_events(
                entity_id
            )

            if not entity_events:
                continue

            # Use a normal event from that user only as the template.
            base_event = random.choice(entity_events)

            timestamp = attack_start + timedelta(
                seconds=offset
            )

            event = self._copy_event(
                base_event,
                timestamp,
            )

            event["source_ip"] = attacker_ip
            event["auth_result"] = "failure"
            event["resource_accessed"] = None
            event["data_downloaded"] = 0
            event["command_sequence"] = []
            event["label"] = "credential_stuffing"

            attack_events.append(event)

        return self.add_events(attack_events)

    # =========================================================
    # 3. LATERAL MOVEMENT
    # =========================================================
    #
    # Short sequence:
    #
    # normal resource
    #       ->
    # unusual resource
    #       ->
    # unusual resource
    #       ->
    # unusual resource
    #
    # Events are separated by only a few minutes.
    # =========================================================

    def add_lateral_movement(self, entity_id):
        entity = self.get_entity(entity_id)

        if entity is None:
            return 0

        duration = self.ATTACK_DURATIONS[
            "lateral_movement"
        ]

        base_event = self._choose_anchor(
            duration,
            entity_id,
        )

        if base_event is None:
            return 0

        number_of_unusual_resources = random.randint(2, 3)

        normal_resources = set(
            entity["normal_resources"]
        )

        unusual_resources = []

        for department, resources in self.resources.items():
            if department == entity["department"]:
                continue

            for resource in resources:
                if resource not in normal_resources:
                    unusual_resources.append(resource)

        if len(unusual_resources) < number_of_unusual_resources:
            return 0

        selected_resources = random.sample(
            unusual_resources,
            number_of_unusual_resources,
        )

        command_sequence = [
            random.choice(
                entity["normal_resources"]
            )
        ]

        command_sequence.extend(
            selected_resources
        )

        # 3-4 events spread over <= 20 minutes.
        sequence_length = len(command_sequence)

        offsets = sorted(
            random.randint(
                10,
                int(duration.total_seconds() - 10),
            )
            for _ in range(sequence_length)
        )

        attack_events = []

        for resource, offset in zip(
            command_sequence,
            offsets,
        ):
            timestamp = (
                base_event["timestamp"]
                + timedelta(seconds=offset)
            )

            event = self._copy_event(
                base_event,
                timestamp,
            )

            event["auth_result"] = "success"
            event["resource_accessed"] = resource
            event["command_sequence"] = command_sequence
            event["data_downloaded"] = 0
            event["label"] = "lateral_movement"

            attack_events.append(event)

        return self.add_events(attack_events)

    # =========================================================
    # 4. IMPOSSIBLE TRAVEL
    # =========================================================
    #
    # Normal location
    #       ->
    # geographically distant location
    #
    # The second event happens shortly after the first one.
    # =========================================================

    def add_impossible_travel(self, entity_id):
        duration = self.ATTACK_DURATIONS[
            "impossible_travel"
        ]

        base_event = self._choose_anchor(
            duration,
            entity_id,
        )

        if base_event is None:
            return 0

        foreign_city = random.choice(
            list(IMPOSSIBLE_TRAVEL_LOCATIONS.keys())
        )

        foreign_location = (
            IMPOSSIBLE_TRAVEL_LOCATIONS[
                foreign_city
            ]
        )

        # Put the anomalous event shortly after the anchor.
        offset = random.randint(
            5 * 60,
            20 * 60,
        )

        event = self._copy_event(
            base_event,
            base_event["timestamp"]
            + timedelta(seconds=offset),
        )

        event["geo_location"] = foreign_location
        event["source_ip"] = fake.ipv4_private()
        event["auth_result"] = "success"
        event["label"] = "impossible_travel"

        return self.add_events([event])

    # =========================================================
    # 5. DEVICE SPOOFING
    # =========================================================
    #
    # Same user
    # very shortly afterward:
    #     new device
    #     new MAC
    #     new source IP
    # =========================================================

    def add_device_spoofing(self, entity_id):
        """
        Device spoofing:

        The attacker is impersonating the user's device.

        Device identity/configuration remains unchanged:
            - device_id
            - mac_address
            - os_version

        The anomaly comes from behavioral/contextual changes:
            - new/suspicious source IP
            - unusual resource
            - abnormal command sequence

        This is intentionally different from:
            - impossible travel: primarily geographic anomaly
            - credential stuffing: cross-user authentication pattern
            - brute force: repeated authentication failures
        """

        duration = self.ATTACK_DURATIONS[
            "device_spoofing"
        ]

        entity = self.get_entity(entity_id)

        if entity is None:
            return 0

        # ---------------------------------------------------------
        # IMPORTANT:
        # We want the user to have an established history before
        # the spoofing event occurs.
        #
        # This makes features such as:
        #     device_seen_before
        #     source_ip_seen_before
        #     resource_seen_before
        #
        # meaningful.
        # ---------------------------------------------------------

        entity_events = self.get_entity_events(
            entity_id
        )

        if not entity_events:
            return 0

        minimum_history = timedelta(
            days=7
        )

        eligible_events = []

        for event in entity_events:

            # Need at least 7 days of normal history
            # before the spoofing event.
            if (
                event["timestamp"] - self.start_time
                >= minimum_history
            ):
                if self._valid_anchor(
                    event["timestamp"],
                    duration,
                    entity_id
                ):
                    eligible_events.append(event)

        if not eligible_events:
            return 0

        base_event = random.choice(
            eligible_events
        )

        # ---------------------------------------------------------
        # Create the spoofing event shortly after the normal event.
        # ---------------------------------------------------------

        timestamp = (
            base_event["timestamp"]
            + timedelta(
                seconds=random.randint(
                    30,
                    int(
                        duration.total_seconds()
                        - 10
                    )
                )
            )
        )

        # Defensive boundary check.
        if not self._valid_anchor(
            base_event["timestamp"],
            duration,
            entity_id
        ):
            return 0

        # ---------------------------------------------------------
        # Find an unusual resource for this user.
        # ---------------------------------------------------------

        normal_resources = set(
            entity["normal_resources"]
        )

        unusual_resources = []

        for department, resources in (
            self.resources.items()
        ):
            for resource in resources:

                if resource not in normal_resources:
                    unusual_resources.append(
                        resource
                    )

        if not unusual_resources:
            return 0

        unusual_resource = random.choice(
            unusual_resources
        )

        # ---------------------------------------------------------
        # Create the spoofed event.
        # ---------------------------------------------------------

        event = self._copy_event(
            base_event,
            timestamp
        )

        # =========================================================
        # DEVICE IDENTITY -- MUST REMAIN THE SAME
        # =========================================================

        event["device_id"] = (
            base_event["device_id"]
        )

        event["mac_address"] = (
            base_event["mac_address"]
        )

        event["os_version"] = (
            base_event["os_version"]
        )

        # =========================================================
        # BEHAVIOR -- THIS IS WHAT CHANGES
        # =========================================================

        # Attacker is coming from a different network context.
        event["source_ip"] = (
            fake.ipv4_private()
        )

        # Keep the location the same so this isn't simply
        # another impossible-travel attack.
        event["geo_location"] = (
            base_event["geo_location"]
        )

        # Authentication succeeds because the attacker has
        # successfully impersonated the device.
        event["auth_result"] = "success"

        # Access something outside the user's normal behavior.
        event["resource_accessed"] = (
            unusual_resource
        )

        # Abnormal command sequence.
        event["command_sequence"] = [
            unusual_resource
        ]

        # No exfiltration is necessary for device spoofing.
        event["data_downloaded"] = 0

        event["label"] = (
            "device_spoofing"
        )

        return self.add_events(
            [event]
        )

    # =========================================================
    # 6. LOW-AND-SLOW EXFILTRATION
    # =========================================================
    #
    # This attack genuinely spans 15 days.
    #
    # Example:
    #
    # Day 0   12 MB
    # Day 3   16 MB
    # Day 6   20 MB
    # Day 10  25 MB
    # Day 15  31 MB
    #
    # The anchor is ONLY selected if all 15 days fit inside
    # the current dataset.
    # =========================================================

    def add_low_slow_exfiltration(self, entity_id):
        duration = self.ATTACK_DURATIONS[
            "low_slow_exfiltration"
        ]

        entity = self.get_entity(entity_id)

        if entity is None:
            return 0

        base_event = self._choose_anchor(
            duration,
            entity_id,
        )

        if base_event is None:
            return 0

        # Fixed minimum span of 15 days, with 5-6 exfiltration events.
        offsets_days = [0, 3, 6, 10, 15]

        # Ensure the final offset is exactly the configured duration.
        offsets = [
            timedelta(days=days)
            for days in offsets_days
        ]

        attack_start = base_event["timestamp"]

        attack_events = []

        current_download = random.uniform(
            10,
            20,
        )

        for i, offset in enumerate(offsets):

            timestamp = attack_start + offset

            # Defensive boundary check.
            if timestamp > self.end_time:
                return 0

            event = self._copy_event(
                base_event,
                timestamp,
            )

            if i > 0:
                current_download += random.uniform(
                    3,
                    8,
                )

            event["data_downloaded"] = round(
                current_download,
                2,
            )

            if event["resource_accessed"] is None:
                event["resource_accessed"] = random.choice(
                    entity["normal_resources"]
                )

            event["auth_result"] = "success"

            if event["resource_accessed"]:
                event["command_sequence"] = [
                    event["resource_accessed"]
                ]

            event["label"] = (
                "low_slow_exfiltration"
            )

            attack_events.append(event)

        return self.add_events(attack_events)

    # =========================================================
    # 7. PRIVILEGE ESCALATION
    # =========================================================
    #
    # Normal user
    #      ->
    # privileged/unusual resource shortly afterward
    # =========================================================

    def add_privilege_escalation(self, entity_id):
        duration = self.ATTACK_DURATIONS[
            "privilege_escalation"
        ]

        entity = self.get_entity(entity_id)

        if entity is None:
            return 0

        base_event = self._choose_anchor(
            duration,
            entity_id,
        )

        if base_event is None:
            return 0

        privileged_resources = []

        for department, resources in self.resources.items():
            if department != entity["department"]:
                privileged_resources.extend(resources)

        if not privileged_resources:
            return 0

        timestamp = (
            base_event["timestamp"]
            + timedelta(
                seconds=random.randint(
                    30,
                    int(duration.total_seconds() - 10),
                )
            )
        )

        privileged_resource = random.choice(
            privileged_resources
        )

        event = self._copy_event(
            base_event,
            timestamp,
        )

        event["auth_result"] = "success"
        event["resource_accessed"] = privileged_resource
        event["command_sequence"] = [
            privileged_resource
        ]
        event["data_downloaded"] = 0
        event["label"] = "privilege_escalation"

        return self.add_events([event])