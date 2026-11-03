# Information Security Policies (Summary)

## DCP-7 Data Classification and Residency Policy
DCP-7 defines four data classes: Public, Internal, Confidential and Restricted. Restricted data must be stored and processed only in EU regions and must be encrypted at rest with customer-managed keys. Any system that stores Restricted data is governed by DCP-7 and must pass an annual residency audit.

## IRP-3 Incident Response Policy
IRP-3 defines incident severities. A P1 incident is any full outage of a Tier 1 system or any confirmed breach of Restricted data. For a P1 incident the on-call engineer must notify the CISO within 30 minutes of detection. Security Operations coordinates the response and a postmortem is required within 5 business days.

## ACP-2 Access Control Policy
Privileged access to production systems is reviewed quarterly by Security Operations. Access not re-certified within 14 days of a review is revoked automatically.
