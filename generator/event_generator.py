import random
from datetime import datetime, timedelta


class EventGenerator:

    def __init__(self, entities, resources):
        self.entities = entities
        self.resources = resources
        self.num_days = 30

    def generate_timestamps(self, entity):
        timestamps = []

        start_date = datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        start_hour = int(entity["working_start_time"].split(":")[0])
        end_hour = int(entity["working_end_time"].split(":")[0])

        for day in range(self.num_days):

            current_day = start_date + timedelta(days=day)

            # 2–6 events per day
            number_of_events = random.randint(2, 6)

            for _ in range(number_of_events):

                hour = random.randint(start_hour, end_hour - 1)
                minute = random.randint(0, 59)

                timestamp = current_day.replace(
                    hour=hour,
                    minute=minute
                )

                timestamps.append(timestamp)

        timestamps.sort()

        return timestamps

    def generate_authentication(self):
        auth_methods = [
            "password",
            "MFA",
            "SSO",
            "biometric"
        ]

        auth_method = random.choice(auth_methods)

        # Small amount of normal authentication failures
        if random.random() < 0.02:
            auth_result = "failure"
        else:
            auth_result = "success"

        return auth_method, auth_result

    # def generate_resource(self, entity):
    #     return random.choice(entity["normal_resources"])

    def generate_events(self):

        events = []

        for entity in self.entities:

            timestamps = self.generate_timestamps(entity)

            for timestamp in timestamps:

                auth_method, auth_result = self.generate_authentication()
                data_downloaded = self.generate_data_downloaded()
                command_sequence = self.generate_command_sequence(entity)
                resource = command_sequence[-1]
                
                event = {
                    "entity_id": entity["entity_id"],
                    "timestamp": timestamp,
                    "source_ip": entity["source_ip"],
                    "geo_location": entity["geo_location"],
                    "auth_method": auth_method,
                    "auth_result": auth_result,
                    "resource_accessed": resource,
                    "device_id": entity["device_id"],
                    "mac_address": entity["mac_address"],
                    "os_version": entity["os_version"],
                    "data_downloaded": data_downloaded,
                    "command_sequence": command_sequence,
                    "label": "normal"
                }

                events.append(event)

        return events

    def generate_data_downloaded(self):
        return round(random.uniform(0.1, 50.0), 2)


    def generate_command_sequence(self, entity):
        resources = entity["normal_resources"]

        sequence_length = random.randint(1, 3)

        return random.sample(
            resources,
            min(sequence_length, len(resources))
        )

