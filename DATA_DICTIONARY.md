# Data dictionary

## products.csv
`product_id` unique ID; `title` listing title; `description` searchable text; `category` product category; `price` INR; `stock` units available; `version` catalogue version; `updated_at` last update timestamp.

## events.csv
`event_id` event ID; `visitor_id` anonymous synthetic visitor key; `event_type` search/view/purchase; `product_id` referenced listing; `timestamp` event time.
