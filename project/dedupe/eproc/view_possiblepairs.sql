CREATE TABLE eprocarp.eprocarp_sbs AS (
SELECT
       ixp,
       ariba_source,
       arp_source,
       arp_target,
       es_score,
       name_source,
       name_target,
       street_source,
       street_target,
       postalcode_source,
       postalcode_target,
       city_source,
       city_target,
       state_source,
       state_target,
       countrycode_source,
       countrycode_target,
       duns_source,
       duns_target,
       cage_source,
       cage_target,
       eu_vat_source,
       eu_vat_target,
       tax1_source,
       tax1_target
FROM (
         SELECT *
         FROM (
                  (SELECT
                          ixp,
                          ariba_source,
                          arp_target,
                          es_score
                   FROM eprocarp.eprocarp_ixp
                  )
              ) p

                  LEFT JOIN
              (SELECT ariba       as ariba_source,
                      name        as name_source,
                      street      as street_source,
                      postalcode  as postalcode_source,
                      city        as city_source,
                      state       AS state_source,
                      countrycode AS countrycode_source,
                      duns        AS duns_source,
                      tax1        AS tax1_source,
                      eu_vat      AS eu_vat_source,
                      cage        as cage_source,
                      arp         AS arp_source
               FROM paco.eproc) as s
              USING (ariba_source)
                  LEFT JOIN
              (SELECT arp         as arp_target,
                      name        as name_target,
                      street      as street_target,
                      postalcode  as postalcode_target,
                      city        as city_target,
                      state       AS state_target,
                      countrycode AS countrycode_target,
                      duns        AS duns_target,
                      eu_vat      AS eu_vat_target,
                      tax1        AS tax1_target,
                      tax2        AS tax2_target,
                      tax3        AS tax3_target,
                      cage        as cage_target
               FROM paco.arp) as t
              USING (arp_target)
     ) c
ORDER BY es_score DESC);

SELECT  * FROM
(SELECT ixp, ariba_source, arp_source, arp_target, concat('ARP_',ariba_source) AS aribaarp, name_source, name_target, ta FROM eprocarp.eprocarp_sbs) b
WHERE arp_source <> arp_target AND aribaarp = arp_target
LIMIT 10;


SELECT * FROM (SELECT arp_target, name_target FROM eprocarp.eprocarp_sbs
WHERE name_target IS null LIMIT 10) s
LEFT JOIN (SELECT arp, name FROM paco.arp) a ON arp_target = arp;

SELECT * FROM paco.arp
WHERE arp = 'ARP_293718';

