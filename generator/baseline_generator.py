from collections import Counter
from statistics import mean, stdev


class BaselineGenerator:

    def __init__(self, events, entities):
        self.events = events
        self.entities = entities

    def generate_user_baselines(self):

        user_baselines = {}

        # --------------------------------------------------
        # Group normal events by entity
        # --------------------------------------------------

        events_by_entity = {}

        for event in self.events:

            entity_id = event["entity_id"]

            if entity_id not in events_by_entity:
                events_by_entity[entity_id] = []

            events_by_entity[entity_id].append(event)

        # --------------------------------------------------
        # Generate baseline for every entity
        # --------------------------------------------------

        for entity in self.entities:

            entity_id = entity["entity_id"]

            user_events = events_by_entity.get(
                entity_id,
                []
            )

            if not user_events:
                continue

            # Make sure events are chronological
            user_events.sort(
                key=lambda event: event["timestamp"]
            )

            # ==================================================
            # STATIC IDENTITY
            # ==================================================

            department = entity["department"]
            role = entity["role"]

            normal_source_ip = entity["source_ip"]
            normal_geo_location = entity["geo_location"]

            normal_device_id = entity["device_id"]
            normal_mac_address = entity["mac_address"]
            normal_os_version = entity["os_version"]

            # ==================================================
            # NORMAL RESOURCES
            # ==================================================

            normal_resources = entity["normal_resources"]

            # ==================================================
            # WORKING TIME
            # ==================================================

            start_hour = int(
                entity["working_start_time"].split(":")[0]
            )

            end_hour = int(
                entity["working_end_time"].split(":")[0]
            )

            # These are static in our synthetic data,
            # but we store them as averages because later
            # they can be updated when we implement drift.

            avg_working_start_hour = start_hour
            avg_working_end_hour = end_hour

            # ==================================================
            # EVENTS PER DAY
            # ==================================================

            unique_dates = set(
                event["timestamp"].date()
                for event in user_events
            )

            number_of_days = len(unique_dates)

            avg_events_per_day = (
                len(user_events) / number_of_days
                if number_of_days > 0
                else 0
            )

            # ==================================================
            # EVENT GAP
            # ==================================================

            # Group timestamps by date so that we don't calculate
            # gaps between the end of one day and the beginning
            # of the next day.

            events_by_date = {}

            for event in user_events:

                event_date = event["timestamp"].date()

                if event_date not in events_by_date:
                    events_by_date[event_date] = []

                events_by_date[event_date].append(
                    event["timestamp"]
                )


            event_gaps = []

            for date, timestamps in events_by_date.items():

                # Sort events within the day
                timestamps.sort()

                # Calculate gaps only within the same day
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

            total_authentications = len(
                user_events
            )

            failed_authentications = sum(
                1
                for event in user_events
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
                for event in user_events
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
            # STORE USER BASELINE
            # ==================================================

            user_baselines[entity_id] = {

                # ------------------------------
                # Identity
                # ------------------------------

                "entity_id": entity_id,
                "department": department,
                "role": role,

                # ------------------------------
                # Network
                # ------------------------------

                "normal_source_ip": normal_source_ip,
                "normal_geo_location": normal_geo_location,

                # ------------------------------
                # Device
                # ------------------------------

                "normal_device_id": normal_device_id,
                "normal_mac_address": normal_mac_address,
                "normal_os_version": normal_os_version,

                # ------------------------------
                # Access
                # ------------------------------

                "normal_resources": normal_resources,

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

        return user_baselines