SELECT
       COUNT(*) as totalpairs,
       SUM(y_true) as totalmatches,
       SUM(y_true)/COUNT(*) as pctage
FROM
     cluster_output;

SELECT
    COUNT(*) as n_samples,
    SUM(y_true)/COUNT(*) as pct,
    AVG(avg_score) as avg_score,
    y_cluster
FROM
    cluster_output
GROUP BY
    y_cluster
ORDER BY avg_score DESC;


WITH possible_false_positives AS (
    SELECT ix, avg_score, cluster_output.y_true
    FROM
    cluster_output
    WHERE
    avg_score>20 AND avg_score <35
    AND cluster_output.y_true = 0)
SELECT
    *
FROM
     possible_false_positives
LEFT JOIN es_sbs USING(ix)
ORDER BY avg_score DESC;

WITH possible_false_negatives AS (
    SELECT ix, avg_score, cluster_output.y_true
    FROM
    cluster_output
    WHERE
    avg_score <20
    AND cluster_output.y_true = 1)
SELECT
    *
FROM
     possible_false_negatives
LEFT JOIN es_sbs USING(ix)
ORDER BY avg_score DESC;
