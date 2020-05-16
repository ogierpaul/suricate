WITH a as (SELECT
       COUNT(*) as total_rows,
       SUM(CASE WHEN (es_score>15) THEN 1 ELSE 0 END) as greater_than_threshold
FROM es_scores)
SELECT
    total_rows,
    greater_than_threshold,
    CAST(greater_than_threshold AS DECIMAL) / total_rows as pct_taken
FROM a

