"""
Mock business data: reservations, properties, smart-lock codes.
In production these would be DB queries / PMS API calls / Seam.co API.
"""
import random
import string


# Mock reservation DB: keyed by booking code (matches @codigo_reserva regex FS\d{6})
RESERVATIONS = {
    "FS123456": {
        "guest_name": "Anto García",
        "guest_name_localized": {"es": "Anto García", "en": "Anto Garcia"},
        "property": "ChampionsGate Villa",
        "check_in": "2026-05-14",
        "check_out": "2026-05-18",
        "guests": 4,
        "status": "active",
    },
    "FS234567": {
        "guest_name": "Sarah Johnson",
        "guest_name_localized": {"es": "Sarah Johnson", "en": "Sarah Johnson"},
        "property": "Reunion Resort",
        "check_in": "2026-05-13",
        "check_out": "2026-05-20",
        "guests": 6,
        "status": "active",
    },
    "FS345678": {
        "guest_name": "Carlos Méndez",
        "guest_name_localized": {"es": "Carlos Méndez", "en": "Carlos Mendez"},
        "property": "Windsor Hills",
        "check_in": "2026-05-15",
        "check_out": "2026-05-22",
        "guests": 3,
        "status": "active",
    },
    "FS456789": {
        "guest_name": "Marina Silva",
        "guest_name_localized": {"es": "Marina Silva", "en": "Marina Silva"},
        "property": "Solterra Resort",
        "check_in": "2026-05-12",
        "check_out": "2026-05-19",
        "guests": 8,
        "status": "active",
    },
}


def lookup_reservation(booking_code: str):
    """Return reservation dict or None if not found."""
    return RESERVATIONS.get(booking_code.upper())


def generate_door_code() -> str:
    """Generate a 6-digit one-time door code (simulates Seam.co / Nuki API call)."""
    return "".join(random.choices(string.digits, k=6))


def open_maintenance_ticket(booking_code: str, issue_summary: str) -> str:
    """Simulate opening a maintenance ticket; return ticket ID."""
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    # In production: POST to ticketing system + WhatsApp alert to Mike
    return ticket_id
