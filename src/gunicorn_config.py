# Gunicorn configuration file
import multiprocessing
 
timeout = 6000
max_requests = 1000
max_requests_jitter = 50
 
log_file = "-"
 
bind = "0.0.0.0:8080"
 
worker_class = "uvicorn.workers.UvicornWorker"
workers = (multiprocessing.cpu_count() * 2) + 1
 
forwarded_allow_ips = "*"
secure_scheme_headers = {"X-Forwarded-Proto": "https"}