# Systems Catalog

This catalog lists production systems, their owners and their dependencies. Ownership determines who remediates defects and incidents.

## Helios Data Platform
Helios is the central lakehouse used for analytics, machine learning and operational reporting. It is owned by the Data Platform Team. Helios runs on infrastructure provided by Stratus Cloud in the eu-west region. Helios stores customer shipment histories, which are classified as Restricted data.

## Customer 360 Dashboard
The Customer 360 Dashboard gives account managers a single view of each customer's shipments, invoices and support tickets. It is built and maintained by the Customer Analytics Team. The dashboard reads all of its data from the Helios Data Platform and refreshes every 15 minutes.

## Atlas Billing Service
Atlas is the invoicing and billing engine. It is owned by Payments Engineering. Atlas stores invoices in the Ledgerline database and sends card and mobile-money payments through QuantaPay. Atlas is classified as a Tier 1 (business-critical) system.

## Ledgerline Database
Ledgerline is a legacy PostgreSQL cluster hosted in the Nairobi data centre. It is owned by Payments Engineering and is scheduled for decommissioning.

## RouteWise Optimizer
RouteWise computes delivery routes for the fleet. It is owned by the Data Platform Team and consumes traffic and shipment data from Helios.
