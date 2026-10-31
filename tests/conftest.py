import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MAUN_DATA_DIR", tempfile.mkdtemp(prefix="maun-test-"))
