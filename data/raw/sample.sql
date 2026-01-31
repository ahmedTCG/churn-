SELECT
  external_customerkey,
  event_time,
  interaction_type,
  incoming_outgoing,
  channel,
  amount,
  shop
FROM poc_dw.customer_interactions_fact
WHERE event_time >= DATEADD(month, -36, CURRENT_DATE)
  AND incoming_outgoing = 'incoming';