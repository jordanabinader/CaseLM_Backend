# Gunicorn configuration file
import multiprocessing
import os
import ssl

timeout = 6000
max_requests = 1000
max_requests_jitter = 50
 
log_file = "-"
 
bind = "0.0.0.0:8000"
 
worker_class = "uvicorn.workers.UvicornWorker"
workers = (multiprocessing.cpu_count() * 2) + 1
 
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}

# SSL configuration
certfile = "cert.pem"
keyfile = "key.pem"

# Make sure the cert files exist
if not os.path.exists(certfile) or not os.path.exists(keyfile):
    raise RuntimeError(f"SSL certificate files not found: {certfile} and {keyfile}")