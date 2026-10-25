import os
import sys
import hashlib
import secrets
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path for configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Try importing pymongo
try:
    import pymongo
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hashes a password with a secure salt using SHA-256."""
    if not salt:
        salt = secrets.token_hex(16)
    salted = f"{salt}{password}".encode('utf-8')
    pw_hash = hashlib.sha256(salted).hexdigest()
    return pw_hash, salt


def verify_password(password: str, pw_hash: str, salt: str) -> bool:
    """Verifies password against stored hash and salt."""
    test_hash, _ = hash_password(password, salt)
    return test_hash == pw_hash


class MongoDBManager:
    """
    MongoDB Atlas Connection & Data Access Manager for Maternal Digital Twin.
    Supports MongoDB Atlas cloud connections with intelligent in-memory fallback.
    """

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or os.getenv("MONGODB_URI", config.MONGODB_URI)
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", config.MONGODB_DB_NAME)
        self.client = None
        self.db = None
        self.is_connected = False
        self.mode = "In-Memory Fallback"

        # In-memory storage structures for fallback mode
        self._memory_users: Dict[str, Dict[str, Any]] = {}
        self._memory_twins: Dict[str, Dict[str, Any]] = {}
        self._memory_vitals: Dict[str, List[Dict[str, Any]]] = {}

        self._connect()
        self._seed_default_demo_data()

    def _connect(self):
        """Attempts connection to MongoDB Atlas."""
        if not PYMONGO_AVAILABLE:
            print("[MongoDBManager] PyMongo not installed. Using In-Memory Storage.")
            self.is_connected = False
            self.mode = "In-Memory Fallback"
            return

        try:
            # Set short timeout so initialization doesn't hang if offline
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            # Send ping to verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.is_connected = True
            self.mode = "MongoDB Atlas"
            print(f"[MongoDBManager] Successfully connected to MongoDB Atlas database: '{self.db_name}'")

            # Create indexes for optimal querying
            self.db[config.MONGO_COLL_USERS].create_index("username", unique=True)
            self.db[config.MONGO_COLL_PATIENT_TWINS].create_index("patient_id", unique=True)
            self.db[config.MONGO_COLL_VITALS_LOGS].create_index([("patient_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])

        except Exception as e:
            print(f"[MongoDBManager] Could not connect to MongoDB Atlas ({e}). Operating in Local In-Memory Fallback mode.")
            self.is_connected = False
            self.mode = "Local In-Memory Fallback"

    def _seed_default_demo_data(self):
        """Seeds default clinician and patient accounts for easy out-of-the-box testing."""
        default_users = [
            {
                "username": "admin",
                "password": "admin123",
                "full_name": "System Administrator",
                "role": "Admin",
                "patient_id": "ADMIN_01",
                "email": "admin@maternaltwin.org"
            },
            {
                "username": "doctor",
                "password": "doctor123",
                "full_name": "Dr. Sarah Jenkins",
                "role": "Clinician",
                "patient_id": "CLINICIAN_01",
                "email": "dr.jenkins@maternaltwin.org"
            },
            {
                "username": "patient001",
                "password": "patient123",
                "full_name": "Emily Watson",
                "role": "Patient",
                "patient_id": "PATIENT_001",
                "email": "emily.w@example.com"
            },
            {
                "username": "patient002",
                "password": "patient123",
                "full_name": "Maria Garcia",
                "role": "Patient",
                "patient_id": "PATIENT_002",
                "email": "maria.g@example.com"
            }
        ]

        for u in default_users:
            if not self.get_user_by_username(u["username"]):
                self.register_user(
                    username=u["username"],
                    password=u["password"],
                    full_name=u["full_name"],
                    role=u["role"],
                    patient_id=u["patient_id"],
                    email=u["email"]
                )

    # --- Authentication & Users ---

    def register_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: str = "Patient",
        patient_id: Optional[str] = None,
        email: str = ""
    ) -> Dict[str, Any]:
        """Registers a new user with hashed password."""
        username = username.strip().lower()
        if not patient_id:
            patient_id = f"PATIENT_{secrets.token_hex(3).upper()}" if role == "Patient" else "CLINICIAN"

        pw_hash, salt = hash_password(password)
        user_doc = {
            "username": username,
            "pw_hash": pw_hash,
            "salt": salt,
            "full_name": full_name,
            "role": role,
            "patient_id": patient_id,
            "email": email,
            "created_at": datetime.now().isoformat()
        }

        if self.is_connected and self.db is not None:
            try:
                self.db[config.MONGO_COLL_USERS].insert_one(user_doc)
            except Exception as e:
                # If unique constraint violation or error
                raise ValueError(f"Username '{username}' already exists or DB error: {e}")
        else:
            if username in self._memory_users:
                raise ValueError(f"Username '{username}' already exists.")
            self._memory_users[username] = user_doc

        # Return clean user profile without hash
        clean_doc = user_doc.copy()
        clean_doc.pop("pw_hash", None)
        clean_doc.pop("salt", None)
        return clean_doc

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verifies user login credentials."""
        username = username.strip().lower()
        user_doc = self.get_user_by_username(username, include_secrets=True)
        if not user_doc:
            return None

        if verify_password(password, user_doc["pw_hash"], user_doc["salt"]):
            clean_doc = user_doc.copy()
            clean_doc.pop("pw_hash", None)
            clean_doc.pop("salt", None)
            clean_doc.pop("_id", None)
            return clean_doc
        return None

    def get_user_by_username(self, username: str, include_secrets: bool = False) -> Optional[Dict[str, Any]]:
        """Finds user profile by username."""
        username = username.strip().lower()
        user_doc = None

        if self.is_connected and self.db is not None:
            user_doc = self.db[config.MONGO_COLL_USERS].find_one({"username": username})
        else:
            user_doc = self._memory_users.get(username)

        if user_doc and not include_secrets:
            user_doc = user_doc.copy()
            user_doc.pop("pw_hash", None)
            user_doc.pop("salt", None)
            user_doc.pop("_id", None)

        return user_doc

    def list_all_patients(self) -> List[Dict[str, Any]]:
        """Lists all registered patients."""
        patients = []
        if self.is_connected and self.db is not None:
            cursor = self.db[config.MONGO_COLL_USERS].find({"role": "Patient"}, {"pw_hash": 0, "salt": 0, "_id": 0})
            patients = list(cursor)
        else:
            patients = [u for u in self._memory_users.values() if u.get("role") == "Patient"]

        # Ensure unique patient IDs list includes PATIENT_001 default
        existing_ids = {p.get("patient_id") for p in patients}
        if "PATIENT_001" not in existing_ids:
            patients.append({
                "patient_id": "PATIENT_001",
                "full_name": "Emily Watson (Default)",
                "username": "patient001",
                "role": "Patient"
            })
        return patients

    def list_all_users(self) -> List[Dict[str, Any]]:
        """Lists all registered users across all roles (Admin, Clinician, Patient)."""
        users = []
        if self.is_connected and self.db is not None:
            cursor = self.db[config.MONGO_COLL_USERS].find({}, {"pw_hash": 0, "salt": 0, "_id": 0})
            users = list(cursor)
        else:
            users = [
                {k: v for k, v in u.items() if k not in ("pw_hash", "salt", "_id")}
                for u in self._memory_users.values()
            ]
        return users

    # --- Digital Twin State Persistence ---

    def save_patient_twin(self, patient_id: str, twin_dict: Dict[str, Any]) -> bool:
        """Upserts Digital Twin state document into MongoDB Atlas."""
        twin_dict["patient_id"] = patient_id
        twin_dict["updated_at"] = datetime.now().isoformat()

        if self.is_connected and self.db is not None:
            try:
                self.db[config.MONGO_COLL_PATIENT_TWINS].replace_one(
                    {"patient_id": patient_id},
                    twin_dict,
                    upsert=True
                )
                return True
            except Exception as e:
                print(f"[MongoDBManager] Error saving twin to Atlas: {e}")
                self._memory_twins[patient_id] = twin_dict
                return False
        else:
            self._memory_twins[patient_id] = twin_dict
            return True

    def get_patient_twin(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves Digital Twin state document from MongoDB Atlas."""
        if self.is_connected and self.db is not None:
            try:
                doc = self.db[config.MONGO_COLL_PATIENT_TWINS].find_one({"patient_id": patient_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                print(f"[MongoDBManager] Error retrieving twin from Atlas: {e}")

        return self._memory_twins.get(patient_id)

    # --- Real-Time Vitals Logging ---

    def log_realtime_vital(self, patient_id: str, vital_data: Dict[str, Any]) -> Dict[str, Any]:
        """Logs real-time vital observation to time-series collection."""
        vital_entry = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "gestational_age_weeks": float(vital_data.get("gestational_age_weeks", 24.0)),
            "systolic_bp": float(vital_data.get("systolic_bp", 120.0)) if vital_data.get("systolic_bp") else None,
            "diastolic_bp": float(vital_data.get("diastolic_bp", 80.0)) if vital_data.get("diastolic_bp") else None,
            "glucose_blood": float(vital_data.get("glucose_blood", 5.5)) if vital_data.get("glucose_blood") else None,
            "glucose_0min": float(vital_data.get("glucose_0min", 5.0)) if vital_data.get("glucose_0min") else None,
            "glucose_120min": float(vital_data.get("glucose_120min", 7.5)) if vital_data.get("glucose_120min") else None,
            "weight_kg": float(vital_data.get("weight_kg", 65.0)) if vital_data.get("weight_kg") else None,
            "bmi": float(vital_data.get("bmi", 25.0)) if vital_data.get("bmi") else None,
            "notes": vital_data.get("notes", "")
        }

        if self.is_connected and self.db is not None:
            try:
                self.db[config.MONGO_COLL_VITALS_LOGS].insert_one(vital_entry)
            except Exception as e:
                print(f"[MongoDBManager] Error inserting vital log to Atlas: {e}")
                if patient_id not in self._memory_vitals:
                    self._memory_vitals[patient_id] = []
                self._memory_vitals[patient_id].append(vital_entry)
        else:
            if patient_id not in self._memory_vitals:
                self._memory_vitals[patient_id] = []
            self._memory_vitals[patient_id].append(vital_entry)

        # Return clean copy without _id
        entry_copy = vital_entry.copy()
        entry_copy.pop("_id", None)
        return entry_copy

    def get_vitals_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Fetches longitudinal time-series vitals history for a patient."""
        history = []
        if self.is_connected and self.db is not None:
            try:
                cursor = self.db[config.MONGO_COLL_VITALS_LOGS].find({"patient_id": patient_id}, {"_id": 0}).sort("timestamp", pymongo.ASCENDING)
                history = list(cursor)
            except Exception as e:
                print(f"[MongoDBManager] Error fetching vitals history: {e}")
                history = self._memory_vitals.get(patient_id, [])
        else:
            history = self._memory_vitals.get(patient_id, [])

        return history

    def get_connection_status(self) -> Dict[str, Any]:
        """Returns connection health and mode status."""
        return {
            "connected": self.is_connected,
            "mode": self.mode,
            "database_name": self.db_name,
            "pymongo_available": PYMONGO_AVAILABLE
        }


# Global Instance
db_manager = MongoDBManager()

# Helper convenience wrappers
def get_connection_status() -> Dict[str, Any]:
    return db_manager.get_connection_status()

def register_user(username, password, full_name, role="Patient", patient_id=None, email="") -> Dict[str, Any]:
    return db_manager.register_user(username, password, full_name, role, patient_id, email)

def authenticate_user(username, password) -> Optional[Dict[str, Any]]:
    return db_manager.authenticate_user(username, password)

def save_patient_twin(patient_id: str, twin_dict: Dict[str, Any]) -> bool:
    return db_manager.save_patient_twin(patient_id, twin_dict)

def get_patient_twin(patient_id: str) -> Optional[Dict[str, Any]]:
    return db_manager.get_patient_twin(patient_id)

def log_realtime_vital(patient_id: str, vital_data: Dict[str, Any]) -> Dict[str, Any]:
    return db_manager.log_realtime_vital(patient_id, vital_data)

def get_vitals_history(patient_id: str) -> List[Dict[str, Any]]:
    return db_manager.get_vitals_history(patient_id)

def list_all_patients() -> List[Dict[str, Any]]:
    return db_manager.list_all_patients()

def list_all_users() -> List[Dict[str, Any]]:
    return db_manager.list_all_users()
