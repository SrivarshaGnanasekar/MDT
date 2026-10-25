"""
Database package for Maternal Digital Twin platform.
Provides MongoDB Atlas connectivity, authentication, patient digital twin state persistence,
and real-time vitals tracking logs.
"""

from database.mongo import (
    db_manager,
    get_connection_status,
    register_user,
    authenticate_user,
    save_patient_twin,
    get_patient_twin,
    log_realtime_vital,
    get_vitals_history,
    list_all_patients,
    list_all_users
)

__all__ = [
    "db_manager",
    "get_connection_status",
    "register_user",
    "authenticate_user",
    "save_patient_twin",
    "get_patient_twin",
    "log_realtime_vital",
    "get_vitals_history",
    "list_all_patients",
    "list_all_users"
]
