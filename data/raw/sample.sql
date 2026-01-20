WITH sampled_customers AS (
  SELECT external_customerkey
  FROM (
    SELECT DISTINCT external_customerkey
    FROM poc_dw.customer_interactions_fact
    WHERE event_time >= DATEADD(month, -24, CURRENT_DATE)
      AND incoming_outgoing = 'incoming'
  ) c
  ORDER BY RANDOM()
  LIMIT 100000
)

SELECT
  external_customerkey,
  event_time,
  interaction_type,
  incoming_outgoing,
  channel,
  amount,
  shop
FROM poc_dw.customer_interactions_fact
WHERE event_time >= DATEADD(month, -12, CURRENT_DATE)
  AND external_customerkey IN (SELECT external_customerkey FROM sampled_customers)
  AND incoming_outgoing = 'incoming';