"""
Retrain and re-validate the early-booking GDM and macrosomia models.

Equivalent to:  python models/train_validated_booking_models.py
Evidence is written to reports/validated/ and the dashboard bundles to
models/validated/.
"""
import time

from models.train_validated_booking_models import main

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nCompleted in {(time.time() - t0) / 60:.1f} minutes. See reports/validated/booking_models_report.txt")
