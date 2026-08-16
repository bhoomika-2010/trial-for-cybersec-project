from datetime import datetime
import math
import ast
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta


def normalize_location(value):
    """Return a 2-tuple coordinate from a CSV string, list, or tuple."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            return None
        return (float(value[0]), float(value[1]))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            parsed = cleaned
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            return (float(parsed[0]), float(parsed[1]))
        if isinstance(parsed, str):
            cleaned_items = parsed.strip("()[] ").split(",")
            if len(cleaned_items) == 2:
                try:
                    return (float(cleaned_items[0]), float(cleaned_items[1]))
                except ValueError:
                    pass
    return None


class FeatureExtractor:

    def __init__(self, events, entities, user_baselines, department_baselines):
        self.events = events
        self.entities = entities
        self.user_baselines = user_baselines
        self.department_baselines = department_baselines

        self.entity_departments = {
            entity["entity_id"]: entity["department"]
            for entity in entities
        }
        # Store events separately for each entity
        self.events_by_entity = {}
        

        for event in self.events:
            entity_id = event["entity_id"]

            if entity_id not in self.events_by_entity:
                self.events_by_entity[entity_id] = []

            self.events_by_entity[entity_id].append(event)

        # Make sure every user's events are chronological
        for entity_id in self.events_by_entity:
            self.events_by_entity[entity_id].sort(
                key=lambda event: event["timestamp"]
            )

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    def get_baseline(self, entity_id):
        """
        Return the user's baseline if available.
        Otherwise fall back to the department baseline.
        """

        user_baseline = self.user_baselines.get(entity_id)

        if user_baseline is not None:
            return user_baseline, "user"

        # Find the user's department
        department = self.entity_departments.get(entity_id)

        if department is None:
            return None, None

        department_baseline = self.department_baselines.get(
            department
        )

        if department_baseline is not None:
            return department_baseline, "department"

        return None, None

    # ---------------------------------------------------------
    # TEMPORAL FEATURES
    # ---------------------------------------------------------

    def get_temporal_features(
        self,
        event,
        previous_event,
        baseline
    ):
        timestamp = event["timestamp"]

        # -----------------------------
        # Time of day
        # -----------------------------

        hour = (
            timestamp.hour
            + timestamp.minute / 60
            + timestamp.second / 3600
        )

        # Cyclic encoding
        hour_sin = math.sin(
            2 * math.pi * hour / 24
        )

        hour_cos = math.cos(
            2 * math.pi * hour / 24
        )

        # -----------------------------
        # Day of week
        # -----------------------------

        day = timestamp.weekday()

        day_sin = math.sin(
            2 * math.pi * day / 7
        )

        day_cos = math.cos(
            2 * math.pi * day / 7
        )

        # -----------------------------
        # Working hours
        # -----------------------------

        working_start = baseline.get(
            "avg_working_start_hour",
            9
        )

        working_end = baseline.get(
            "avg_working_end_hour",
            18
        )

        outside_working_hours = int(
            hour < working_start
            or hour > working_end
        )

        # -----------------------------
        # Time since previous event
        # -----------------------------

        if previous_event is None:

            time_since_previous_event = 0.0

        else:

            time_difference = (
                timestamp
                - previous_event["timestamp"]
            )

            time_since_previous_event = (
                time_difference.total_seconds()
                / 60
            )

        # -----------------------------
        # Event-gap z-score
        # -----------------------------

        baseline_avg_gap = baseline.get(
            "avg_event_gap_minutes"
        )

        baseline_std_gap = baseline.get(
            "std_event_gap_minutes"
        )

        if (
            baseline_avg_gap is not None
            and baseline_std_gap is not None
            and baseline_std_gap > 0
            and previous_event is not None
        ):

            event_gap_zscore = (
                time_since_previous_event
                - baseline_avg_gap
            ) / baseline_std_gap

        else:

            event_gap_zscore = 0.0

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "outside_working_hours": outside_working_hours,
            "time_since_previous_event": time_since_previous_event,
            "event_gap_zscore": event_gap_zscore
        }

    # ---------------------------------------------------------
    # ONE EVENT
    # ---------------------------------------------------------

    def extract_event_features(
        self,
        event,
        previous_event=None,
        current_index=None
    ):

        entity_id = event["entity_id"]

        baseline, baseline_type = self.get_baseline(
            entity_id
        )

        if baseline is None:
            raise ValueError(
                f"No baseline found for {entity_id}"
            )
        entity_events = self.events_by_entity[entity_id]

        if current_index is None:
            current_index = entity_events.index(event)

        temporal_features = self.get_temporal_features(
            event,
            previous_event,
            baseline
        )
        authentication_features = (
            self.get_authentication_features(
                event,
                entity_events,
                current_index
            )
        )
        network_device_features = (
            self.get_network_device_features(
                event,
                entity_events,
                current_index
            )
        )
        geography_features = (
            self.get_geography_features(
                event,
                previous_event,
                entity_events,
                current_index
            )
        )
        resource_features = (
            self.get_resource_features(
                event,
                entity_events,
                current_index
            )
        )
        data_activity_features = (
            self.get_data_activity_features(
                event,
                entity_events,
                current_index
            )
        )
        cross_user_auth_features = (
            self.get_cross_user_auth_features(
                event
            )
        )

        features = {
            **temporal_features,
            **authentication_features,
            **network_device_features,
            **geography_features,
            **resource_features,
            **data_activity_features,
            **cross_user_auth_features,
            "baseline_type": (
                1 if baseline_type == "user"
                else 0
            )
        }

        return features

    def get_authentication_features(self, event, entity_events, current_index):
        auth_failed = int(
            event["auth_result"].lower() == "failure"
        )

        auth_method = event["auth_method"]

        # ---------------------------------
        # Recent events
        # ---------------------------------

        recent_events = entity_events[
            max(0, current_index - 9): current_index + 1
        ]

        # ---------------------------------
        # Recent authentication failures
        # ---------------------------------

        recent_failures = sum(
            1
            for e in recent_events
            if e["auth_result"].lower() == "failure"
        )

        recent_auth_failure_rate = (
            recent_failures / len(recent_events)
            if recent_events
            else 0.0
        )

        # ---------------------------------
        # Consecutive failures
        # ---------------------------------

        consecutive_auth_failures = 0

        for i in range(current_index, -1, -1):

            if entity_events[i]["auth_result"].lower() == "failure":
                consecutive_auth_failures += 1
            else:
                break

        return {
            "auth_failed": auth_failed,
            "auth_method": auth_method,
            "consecutive_auth_failures": consecutive_auth_failures,
            "recent_auth_failure_rate": recent_auth_failure_rate
        }

    def get_network_device_features(
        self,
        event,
        entity_events,
        current_index
    ):
        # ---------------------------------
        # Current user's normal identity
        # ---------------------------------

        entity_id = event["entity_id"]

        baseline, _ = self.get_baseline(entity_id)

        normal_ip = baseline.get(
            "normal_source_ip"
        )

        normal_device = baseline.get(
            "normal_device_id"
        )

        normal_mac = baseline.get(
            "normal_mac_address"
        )

        normal_os = baseline.get(
            "normal_os_version"
        )

        # ---------------------------------
        # Source IP
        # ---------------------------------

        source_ip_changed = int(
            event["source_ip"] != normal_ip
        )

        previous_events = entity_events[
            :current_index
        ]

        source_ip_seen_before = int(
            event["source_ip"]
            in {
                e["source_ip"]
                for e in previous_events
            }
        )

        # ---------------------------------
        # Recent unique source IPs
        # ---------------------------------

        recent_events = entity_events[
            max(0, current_index - 9):
            current_index + 1
        ]

        unique_source_ips_recent = len(
            {
                e["source_ip"]
                for e in recent_events
            }
        )

        # ---------------------------------
        # Device
        # ---------------------------------

        device_changed = int(
            event["device_id"] != normal_device
        )

        device_seen_before = int(
            event["device_id"]
            in {
                e["device_id"]
                for e in previous_events
            }
        )

        # ---------------------------------
        # MAC address
        # ---------------------------------

        mac_changed = int(
            event["mac_address"] != normal_mac
        )

        # ---------------------------------
        # Operating system
        # ---------------------------------

        os_changed = int(
            event["os_version"] != normal_os
        )

        return {
            "source_ip_changed": source_ip_changed,
            "source_ip_seen_before": source_ip_seen_before,
            "unique_source_ips_recent": unique_source_ips_recent,

            "device_changed": device_changed,
            "device_seen_before": device_seen_before,

            "mac_changed": mac_changed,
            "os_changed": os_changed
        }

    def calculate_distance_km(self, location1, location2):
            """
            Calculate the great-circle distance between two
            latitude/longitude coordinates using the Haversine formula.
            """

            location1 = normalize_location(location1)
            location2 = normalize_location(location2)

            if location1 is None or location2 is None:
                return 0.0

            lat1, lon1 = location1
            lat2, lon2 = location2

            lat1 = radians(lat1)
            lon1 = radians(lon1)

            lat2 = radians(lat2)
            lon2 = radians(lon2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                sin(dlat / 2) ** 2
                + cos(lat1)
                * cos(lat2)
                * sin(dlon / 2) ** 2
            )

            c = 2 * atan2(
                sqrt(a),
                sqrt(1 - a)
            )

            earth_radius_km = 6371

            return earth_radius_km * c

    def get_geography_features(
        self,
        event,
        previous_event,
        entity_events,
        current_index
    ):

        entity_id = event["entity_id"]

        baseline, _ = self.get_baseline(
            entity_id
        )

        current_location = normalize_location(
            event.get("geo_location")
        )

        normal_location = normalize_location(
            baseline.get("normal_geo_location")
        )

        # ---------------------------------
        # Location changed from home
        # ---------------------------------

        geo_changed = int(
            current_location != normal_location
        )

        # ---------------------------------
        # Has this location been seen before?
        # ---------------------------------

        previous_events = entity_events[
            :current_index
        ]

        previous_locations = {
            tuple(normalize_location(e.get("geo_location")) or ())
            for e in previous_events
        }

        location_seen_before = int(
            tuple(current_location or ())
            in previous_locations
        )

        # ---------------------------------
        # Distance from user's normal
        # location
        # ---------------------------------

        if normal_location is not None:

            distance_from_home = (
                self.calculate_distance_km(
                    normal_location,
                    current_location
                )
            )

        else:

            distance_from_home = 0.0

        # ---------------------------------
        # Distance from previous event
        # ---------------------------------

        if previous_event is None:

            distance_from_previous = 0.0
            travel_speed = 0.0

        else:

            previous_location = normalize_location(
                previous_event.get("geo_location")
            )

            distance_from_previous = (
                self.calculate_distance_km(
                    previous_location,
                    current_location
                )
            )

            time_difference = (
                event["timestamp"]
                - previous_event["timestamp"]
            )

            time_hours = (
                time_difference.total_seconds()
                / 3600
            )

            if time_hours > 0:

                travel_speed = (
                    distance_from_previous
                    / time_hours
                )

            else:

                travel_speed = 0.0

        return {
            "geo_changed": geo_changed,
            "location_seen_before": location_seen_before,
            "distance_from_home": distance_from_home,
            "distance_from_previous": distance_from_previous,
            "travel_speed": travel_speed
        }

    def get_resource_features(
        self,
        event,
        entity_events,
        current_index
    ):

        entity_id = event["entity_id"]

        baseline, _ = self.get_baseline(
            entity_id
        )

        normal_resources = set(
            baseline.get(
                "normal_resources",
                []
            )
        )

        current_resource = event[
            "resource_accessed"
        ]

        # ---------------------------------
        # Is current resource normal?
        # ---------------------------------

        resource_is_normal = int(
            current_resource in normal_resources
        )

        resource_is_anomalous = int(
            current_resource not in normal_resources
        )

        # ---------------------------------
        # Has this resource been seen
        # before by this user?
        # ---------------------------------

        previous_events = entity_events[
            :current_index
        ]

        previous_resources = {
            e["resource_accessed"]
            for e in previous_events
        }

        resource_seen_before = int(
            current_resource in previous_resources
        )

        # ---------------------------------
        # Number of unique resources
        # recently accessed
        # ---------------------------------

        recent_events = entity_events[
            max(0, current_index - 9):
            current_index + 1
        ]

        recent_resources = {
            e["resource_accessed"]
            for e in recent_events
        }

        unique_resources_recent = len(
            recent_resources
        )

        # ---------------------------------
        # Number of normal resources
        # accessed recently
        # ---------------------------------

        normal_resources_recent = len(
            recent_resources.intersection(
                normal_resources
            )
        )

        # ---------------------------------
        # Number of abnormal resources
        # accessed recently
        # ---------------------------------

        abnormal_resources_recent = len(
            recent_resources - normal_resources
        )

        # ---------------------------------
        # Resources outside normal set
        # in current event's command sequence
        # ---------------------------------

        command_sequence = event.get(
            "command_sequence",
            []
        )

        abnormal_commands = [
            resource
            for resource in command_sequence
            if resource not in normal_resources
        ]

        abnormal_command_count = len(
            abnormal_commands
        )

        return {
            "resource_is_normal": resource_is_normal,
            "resource_is_anomalous": resource_is_anomalous,
            "resource_seen_before": resource_seen_before,
            "unique_resources_recent": unique_resources_recent,
            "normal_resources_recent": normal_resources_recent,
            "abnormal_resources_recent": abnormal_resources_recent,
            "abnormal_command_count": abnormal_command_count
        }

    def get_data_activity_features(
        self,
        event,
        entity_events,
        current_index
    ):

        entity_id = event["entity_id"]

        baseline, _ = self.get_baseline(entity_id)

        current_download = float(
            event.get("data_downloaded", 0)
        )

        avg_download = float(
            baseline.get("avg_data_downloaded", 0)
        )

        std_download = float(
            baseline.get("std_data_downloaded", 0)
        )

        # =====================================================
        # 1. Individual download deviation
        # =====================================================

        if std_download > 0:
            download_zscore = (
                current_download - avg_download
            ) / std_download
        else:
            download_zscore = 0.0

        if avg_download > 0:
            download_ratio = (
                current_download / avg_download
            )
        else:
            download_ratio = 0.0

        # =====================================================
        # Only events up to the current event
        # =====================================================

        previous_events = entity_events[
            :current_index + 1
        ]

        current_timestamp = event["timestamp"]

        # =====================================================
        # Rolling-window helper
        # =====================================================

        def events_in_window(days):

            start_time = (
                current_timestamp
                - timedelta(days=days)
            )

            return [
                e
                for e in previous_events
                if start_time <= e["timestamp"]
                <= current_timestamp
            ]

        # =====================================================
        # Build windows
        # =====================================================

        events_3d = events_in_window(3)
        events_7d = events_in_window(7)
        events_15d = events_in_window(15)

        # =====================================================
        # Total downloads
        # =====================================================

        download_3d = sum(
            float(e.get("data_downloaded", 0))
            for e in events_3d
        )

        download_7d = sum(
            float(e.get("data_downloaded", 0))
            for e in events_7d
        )

        download_15d = sum(
            float(e.get("data_downloaded", 0))
            for e in events_15d
        )

        # =====================================================
        # Event counts
        # =====================================================

        events_count_3d = len(events_3d)
        events_count_7d = len(events_7d)
        events_count_15d = len(events_15d)

        # =====================================================
        # Average download per observed event
        #
        # This avoids assuming that every user generated
        # exactly the baseline average number of events.
        # =====================================================

        if events_count_3d > 0:

            avg_download_per_event_3d = (
                download_3d / events_count_3d
            )

        else:

            avg_download_per_event_3d = 0.0

        if events_count_7d > 0:

            avg_download_per_event_7d = (
                download_7d / events_count_7d
            )

        else:

            avg_download_per_event_7d = 0.0

        if events_count_15d > 0:

            avg_download_per_event_15d = (
                download_15d / events_count_15d
            )

        else:

            avg_download_per_event_15d = 0.0

        # =====================================================
        # Cumulative excess download over 15 days
        #
        # Expected download is based on the ACTUAL number
        # of events, not expected events per day.
        # =====================================================

        expected_download_15d = (
            avg_download
            * events_count_15d
        )

        cumulative_excess_download_15d = (
            download_15d
            - expected_download_15d
        )

        # =====================================================
        # Daily excess download trend
        #
        # For every day:
        #
        # actual daily download
        # -
        # expected daily download
        #
        # Then calculate the slope.
        # =====================================================

        daily_downloads = {}

        for e in events_15d:

            date = e["timestamp"].date()

            daily_downloads.setdefault(
                date,
                0.0
            )

            daily_downloads[date] += float(
                e.get("data_downloaded", 0)
            )

        daily_excess_points = []

        for date, total_download in (
            daily_downloads.items()
        ):

            day_event_count = sum(
                1
                for e in events_15d
                if e["timestamp"].date() == date
            )

            expected_daily_download = (
                avg_download
                * day_event_count
            )

            daily_excess = (
                total_download
                - expected_daily_download
            )

            day_number = (
                date
                - current_timestamp.date()
            ).days

            daily_excess_points.append(
                (
                    day_number,
                    daily_excess
                )
            )

        daily_excess_download_trend_15d = 0.0

        if len(daily_excess_points) >= 2:

            x_values = [
                point[0]
                for point in daily_excess_points
            ]

            y_values = [
                point[1]
                for point in daily_excess_points
            ]

            x_mean = (
                sum(x_values)
                / len(x_values)
            )

            y_mean = (
                sum(y_values)
                / len(y_values)
            )

            numerator = sum(
                (x - x_mean)
                * (y - y_mean)
                for x, y in zip(
                    x_values,
                    y_values
                )
            )

            denominator = sum(
                (x - x_mean) ** 2
                for x in x_values
            )

            if denominator > 0:

                daily_excess_download_trend_15d = (
                    numerator / denominator
                )

        # =====================================================
        # Return features
        # =====================================================

        return {

            "data_downloaded":
                current_download,

            "download_zscore":
                download_zscore,

            "download_ratio":
                download_ratio,

            "download_3d":
                download_3d,

            "download_7d":
                download_7d,

            "download_15d":
                download_15d,

            "avg_download_per_event_3d":
                avg_download_per_event_3d,

            "avg_download_per_event_7d":
                avg_download_per_event_7d,

            "avg_download_per_event_15d":
                avg_download_per_event_15d,

            "cumulative_excess_download_15d":
                cumulative_excess_download_15d,

            "daily_excess_download_trend_15d":
                daily_excess_download_trend_15d,

            "events_count_3d":
                events_count_3d,

            "events_count_7d":
                events_count_7d,

            "events_count_15d":
                events_count_15d
        }

    def get_cross_user_auth_features(
        self,
        event
    ):

        current_timestamp = event["timestamp"]
        current_source_ip = event["source_ip"]

        # -----------------------------------------------------
        # Look at authentication activity from this source IP
        # during the recent time window.
        #
        # 15 minutes is appropriate because credential
        # stuffing is a burst of attempts across accounts.
        # -----------------------------------------------------

        window_start = (
            current_timestamp
            - timedelta(minutes=15)
        )

        recent_source_events = [
            e
            for e in self.events
            if (
                e["timestamp"] >= window_start
                and e["timestamp"] <= current_timestamp
                and e["source_ip"] == current_source_ip
                and e.get("auth_result") is not None
            )
        ]

        # -----------------------------------------------------
        # Total authentication attempts from this IP
        # -----------------------------------------------------

        source_ip_auth_attempts_recent = len(
            recent_source_events
        )

        # -----------------------------------------------------
        # Authentication failures from this IP
        # -----------------------------------------------------

        failed_events = [
            e
            for e in recent_source_events
            if e.get("auth_result") == "failure"
        ]

        source_ip_auth_failures_recent = len(
            failed_events
        )

        # -----------------------------------------------------
        # Distinct users targeted by this IP
        # -----------------------------------------------------

        targeted_users = {
            e["entity_id"]
            for e in recent_source_events
        }

        source_ip_target_count = len(
            targeted_users
        )

        # -----------------------------------------------------
        # Distinct users that experienced failures
        # -----------------------------------------------------

        failed_users = {
            e["entity_id"]
            for e in failed_events
        }

        source_ip_failed_users_recent = len(
            failed_users
        )

        # -----------------------------------------------------
        # Failure rate of this source IP
        # -----------------------------------------------------

        if source_ip_auth_attempts_recent > 0:

            source_ip_failure_rate = (
                source_ip_auth_failures_recent
                / source_ip_auth_attempts_recent
            )

        else:

            source_ip_failure_rate = 0.0

        # -----------------------------------------------------
        # What fraction of targeted users experienced
        # authentication failures?
        # -----------------------------------------------------

        if source_ip_target_count > 0:

            source_ip_failed_user_ratio = (
                source_ip_failed_users_recent
                / source_ip_target_count
            )

        else:

            source_ip_failed_user_ratio = 0.0

        # -----------------------------------------------------
        # How widely is this IP spreading authentication
        # attempts across different users?
        #
        # High value:
        #   many different users targeted
        #
        # Low value:
        #   same user targeted repeatedly
        # -----------------------------------------------------

        if source_ip_auth_attempts_recent > 0:

            source_ip_target_spread_rate = (
                source_ip_target_count
                / source_ip_auth_attempts_recent
            )

        else:

            source_ip_target_spread_rate = 0.0

        return {

            "source_ip_auth_attempts_recent":
                source_ip_auth_attempts_recent,

            "source_ip_auth_failures_recent":
                source_ip_auth_failures_recent,

            "source_ip_target_count":
                source_ip_target_count,

            "source_ip_failed_users_recent":
                source_ip_failed_users_recent,

            "source_ip_failure_rate":
                source_ip_failure_rate,

            "source_ip_failed_user_ratio":
                source_ip_failed_user_ratio,

            "source_ip_target_spread_rate":
                source_ip_target_spread_rate
        }