import random
from faker import Faker
CITY_LOCATIONS = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867)
}
fake = Faker()
class Office:
    def __init__(self):
        self.departments = {
            "IT": [
                "Software Developer",
                "System Administrator",
                "DevOps Engineer",
                "IT Support",
                "Security Analyst"
            ],
            "Finance": [
                "Accountant",
                "Financial Analyst",
                "Finance Manager"
            ],
            "HR": [
                "HR Executive",
                "Recruiter",
                "HR Manager"
            ],
            "Sales": [
                "Sales Executive",
                "Sales Analyst",
                "Sales Manager"
            ],
            "Marketing": [
                "Marketing Executive",
                "Content Specialist",
                "Marketing Manager"
            ],
            "Operations": [
                "Operations Executive",
                "Operations Analyst",
                "Operations Manager"
            ]
        }
        self.resources = {
            "IT": [
                "Code Repository",
                "Development Server",
                "Monitoring System",
                "IT Helpdesk",
                "Documentation Portal"
            ],

            "Finance": [
                "Accounting System",
                "Payroll System",
                "Financial Reports",
                "Invoice System",
                "Banking Portal"
            ],

            "HR": [
                "Employee Database",
                "Recruitment System",
                "HR Documents",
                "Payroll Portal",
                "Leave Management System"
            ],

            "Sales": [
                "CRM",
                "Sales Dashboard",
                "Customer Database",
                "Sales Reports",
                "Quotation System"
            ],

            "Marketing": [
                "Marketing Dashboard",
                "Campaign Management System",
                "Content Repository",
                "Analytics Portal",
                "Social Media Manager"
            ],

            "Operations": [
                "Operations Dashboard",
                "Inventory System",
                "Order Management System",
                "Operations Reports",
                "Logistics Portal"
            ]
        }

        self.home_locations=["Pune", "Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad"]

        self.department_sizes = {
            "IT": 20,
            "Finance": 15,
            "HR": 10,
            "Sales": 20,
            "Marketing": 15,
            "Operations": 20
        }

        self.os_versions = [
            "Windows 11",
            "Windows 10",
            "macOS",
            "Ubuntu Linux"
        ]

    def generate_resource_footprint(self, department):
        resources = self.resources[department]

        footprint_size = random.randint(2, len(resources))

        return random.sample(resources, footprint_size)
    def generate_entities(self):
        entities = []

        entity_id = 1

        for department, count in self.department_sizes.items():
            roles = self.departments[department]
            

            for _ in range(count):
                role = random.choice(roles)
                home_location = random.choice(self.home_locations)
                normal_resources = self.generate_resource_footprint(department)
                geo_location = CITY_LOCATIONS[home_location]
                start_hour = random.randint(8, 10)
                end_hour = random.randint(17, 19)
                device_id = f"DEV{entity_id:03d}"
                entity = {
                    "entity_id": f"E{entity_id:03d}",
                    "entity_type": "user",
                    "department": department,
                    "role": role,
                    "home_location": home_location,
                    "working_start_time": f"{start_hour:02d}:00",
                    "working_end_time": f"{end_hour:02d}:00",
                    "source_ip": fake.ipv4_private(),
                    "geo_location" : geo_location,
                    "device_id": device_id,
                    "os_version": random.choice(self.os_versions),
                    "mac_address": fake.mac_address(),
                    "normal_resources": normal_resources,
                }

                entities.append(entity)
                entity_id += 1

        return entities