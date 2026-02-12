# DDoS Traffic Analysis

## Traffic Anomaly Detection

The traffic analysis reveals a significant anomaly between 18:24 and 18:29, where the number of requests per minute sharply increases from a baseline of approximately 500–1500 requests to over 12,000 requests per minute.

This abnormal surge indicates a potential Distributed Denial-of-Service (DDoS) attack.

The presence of multiple source IP addresses and repeated requests to sensitive endpoints such as /usr/admin and /usr/login further supports the likelihood of coordinated malicious activity.

## Requests per Minute
![Requests per Minute](requests_per_minute.png)

## HTTP Status Code Distribution
![Status Codes](status_codes.png)

## Top Requested Endpoints
![Top Endpoints](top_endpoints.png)

## Top IP Addresses
![Top IPs](top_ips.png)


Impact Assessment

The observed traffic spike could significantly impact server availability, performance, and reliability. A sudden increase to over 12,000 requests per minute may exhaust system resources such as CPU, memory, and network bandwidth, potentially causing service degradation or downtime.

Security Implications

The distribution of HTTP status codes, including a high number of 4xx and 5xx responses, suggests abnormal client behavior and possible exploitation attempts. Repeated access to administrative and authentication-related endpoints increases the risk of brute-force attacks, credential stuffing, or vulnerability probing.

Recommendations

To mitigate similar incidents, it is recommended to implement rate limiting, Web Application Firewall (WAF) protection, IP reputation filtering, and real-time monitoring solutions. Additionally, enabling detailed logging and automated anomaly detection mechanisms can help identify and respond to DDoS attacks more efficiently in the future.
