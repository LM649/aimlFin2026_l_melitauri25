import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

log_file = "l_melitauri25_31698_server.log"

data = []

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            parts = line.split()

            ip = parts[0]

            # Timestamp
            start = line.find("[")
            end = line.find("]")
            time_str = line[start + 1:end]

            dt = datetime.strptime(time_str.split("+")[0], "%Y-%m-%d %H:%M:%S")

            method = parts[5].replace('"', '')
            endpoint = parts[6]
            status = int(parts[8])

            data.append([ip, dt, method, endpoint, status])
        except:
            continue

df = pd.DataFrame(data, columns=["ip", "timestamp", "method", "endpoint", "status"])

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna()

# -----------------------
# 1️⃣ Requests per minute
# -----------------------
df["minute"] = df["timestamp"].dt.floor("min")
rpm = df.groupby("minute").size()

plt.figure(figsize=(12, 6))
plt.plot(rpm)
plt.title("Requests per Minute")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("requests_per_minute.png")

# -----------------------
# 2️⃣ Top IPs
# -----------------------
top_ips = df["ip"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_ips.plot(kind="bar")
plt.title("Top 10 IP Addresses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_ips.png")

# -----------------------
# 3️⃣ Status Codes
# -----------------------
status_counts = df["status"].value_counts()

plt.figure(figsize=(8, 6))
status_counts.plot(kind="bar")
plt.title("HTTP Status Codes Distribution")
plt.tight_layout()
plt.savefig("status_codes.png")

# -----------------------
# 4️⃣ Top Endpoints
# -----------------------
top_endpoints = df["endpoint"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_endpoints.plot(kind="bar")
plt.title("Top 10 Requested Endpoints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_endpoints.png")
