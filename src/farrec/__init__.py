"""FAR-Trans investment recommender."""
import os

# implicit (ALS) performs badly with a multithreaded OpenBLAS pool; must be set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
