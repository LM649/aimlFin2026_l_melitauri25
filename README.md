Conclusion

The conducted traffic analysis identified a clear abnormal spike in request volume within a short time window. The number of requests per minute increased dramatically compared to the normal baseline traffic pattern.

Such behavior is highly indicative of a potential Distributed Denial-of-Service (DDoS) attack, where multiple source IP addresses generate excessive traffic targeting specific endpoints.

Further security investigation is recommended, including IP reputation analysis, rate-limiting implementation, and web application firewall (WAF) configuration to mitigate similar attacks in the future.

![Requests per Minute](requests_per_minute.png)

![Status Codes](status_codes.png)

![Top Endpoints](top_endpoints.png)

![Top IPs](top_ips.png)
