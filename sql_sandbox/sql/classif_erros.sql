SELECT * FROM sbs_matches
WHERE
y_true = 1
AND
y_pred_rf = 0
ORDER BY name_vecchar DESC
LIMIT 10;
