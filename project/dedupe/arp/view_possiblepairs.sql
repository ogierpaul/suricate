CREATE TABLE arp_sbs AS
SELECT ixp,
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
       arp_harmonizedname_source,
       arp_harmonizedname_target,
       arp_partnercompany_source,
       arp_partnercompany_target,
       eu_vat_source,
       eu_vat_target,
       tax1_source,
       tax1_target,
       tax2_source,
       tax2_target,
       tax3_source,
       tax3_target,
       concatenatedids_source,
       concatenatedids_target
FROM (
         SELECT *
         FROM (
                  (SELECT ixp,
                          arp_source,
                          arp_target,
                          es_score
                   FROM arp_ixp
                  )
              ) p

                  LEFT JOIN
              (SELECT arp                as arp_source,
                      name               as name_source,
                      street             as street_source,
                      postalcode         as postalcode_source,
                      city               as city_source,
                      state              AS state_source,
                      countrycode        AS countrycode_source,
                      duns               AS duns_source,
                      tax1               AS tax1_source,
                      tax2               AS tax2_source,
                      tax3               AS tax3_source,
                      eu_vat             AS eu_vat_source,
                      concatenatedids    as concatenatedids_source,
                      arp_harmonizedname as arp_harmonizedname_source,
                      arp_partnercompany AS arp_partnercompany_source,
                      cage               as cage_source
               FROM arp) as s
              USING (arp_source)
                  LEFT JOIN
              (SELECT arp                as arp_target,
                      name               as name_target,
                      street             as street_target,
                      postalcode         as postalcode_target,
                      city               as city_target,
                      state              AS state_target,
                      countrycode        AS countrycode_target,
                      duns               AS duns_target,
                      concatenatedids    as concatenatedids_target,
                      arp_harmonizedname as arp_harmonizedname_target,
                      arp_partnercompany AS arp_partnercompany_target,
                      eu_vat             AS eu_vat_target,
                      tax1               AS tax1_target,
                      tax2               AS tax2_target,
                      tax3               AS tax3_target,
                      cage               as cage_target
               FROM arp) as t
              USING (arp_target)
     ) c
    WITH DATA
