import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from database.mongo import (
    db_manager,
    register_user,
    authenticate_user,
    save_patient_twin,
    get_patient_twin,
    log_realtime_vital,
    get_vitals_history,
    list_all_patients
)

def run_tests():
    print("=== TESTING MONGODB ATLAS INTEGRATION ===")
    status = db_manager.get_connection_status()
    print(f"DB Status: {status}")

    # Test User Registration & Auth
    print("\n1. Testing Auth & User Registration:")
    try:
        user = register_user("patient_jane", "jane123", "Jane Doe", role="Patient", patient_id="PATIENT_JANE")
        print(f"Registered User: {user['username']} | Patient ID: {user['patient_id']}")
    except ValueError as e:
        print(f"User registration note: {e}")

    authed = authenticate_user("patient_jane", "jane123")
    assert authed is not None, "Authentication failed for registered user!"
    print(f"Authenticated successfully as: {authed['full_name']} ({authed['role']})")

    # Test Vitals Logging
    print("\n2. Testing Real-Time Vitals Logging:")
    vital_entry = log_realtime_vital("PATIENT_JANE", {
        "gestational_age_weeks": 26.0,
        "systolic_bp": 122.0,
        "diastolic_bp": 80.0,
        "glucose_blood": 5.8,
        "glucose_120min": 7.9,
        "weight_kg": 68.5,
        "notes": "Patient logged morning vitals via web portal."
    })
    print(f"Logged vital at {vital_entry['timestamp']} for {vital_entry['patient_id']}")

    history = get_vitals_history("PATIENT_JANE")
    print(f"Total vitals logged for PATIENT_JANE: {len(history)}")
    assert len(history) > 0, "Vitals history should not be empty!"

    # Test Twin State Persistence
    print("\n3. Testing Digital Twin State Persistence:")
    twin_data = {
        "patient_id": "PATIENT_JANE",
        "maternal_state": {"age": 29, "bmi": 26.4},
        "fetal_state": {"fetal_ac_cm": 22.1},
        "recommendations": ["Maintain current diet", "Schedule next ultrasound at week 28"]
    }
    saved = save_patient_twin("PATIENT_JANE", twin_data)
    print(f"Saved twin to Atlas/Fallback: {saved}")

    loaded_twin = get_patient_twin("PATIENT_JANE")
    assert loaded_twin is not None, "Twin state should be retrievable!"
    print(f"Retrieved Digital Twin for: {loaded_twin['patient_id']} | Recommendations: {loaded_twin['recommendations']}")

    print("\n4. Testing Clinician Patient Listing:")
    patients = list_all_patients()
    print(f"Registered patients count: {len(patients)}")
    for p in patients:
        print(f" - {p.get('full_name')} ({p.get('patient_id')})")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
