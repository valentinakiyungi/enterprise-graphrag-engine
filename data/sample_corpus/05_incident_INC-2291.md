# Postmortem: INC-2291 — eu-west Storage Outage

- Date: 14 May 2026
- Severity: P2
- Duration: 4 hours 10 minutes (09:05–13:15 EAT)

## Summary
A storage-control-plane failure in Stratus Cloud's eu-west region made object storage unavailable. Helios ingestion and query jobs failed for the duration of the outage. Downstream consumers served stale data; account managers reported that customer views were up to four hours out of date.

## Impact
- Analytics queries failed for 4 hours 10 minutes.
- RouteWise fell back to cached routes, increasing average delivery time by 6%.
- No data was lost and no Restricted data was exposed.

## Remediation
Remediation is owned by the team that owns the affected platform. Actions:
1. Add cross-region replication for critical Helios tables (due 30 June 2026).
2. Request a service credit from the vendor for the availability breach.
3. Add staleness banners to downstream dashboards.
