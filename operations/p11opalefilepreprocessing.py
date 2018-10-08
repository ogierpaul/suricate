# newright = pd.read_excel(
#     filepath_right
# ).rename(
#     columns={
#         'supplierid': ixname,
#         'country_code': 'countrycode',
#         'eu_vat': 'euvat'
#     }
# ).drop(
#     [
#         'Code\nPost',
#         'Dept',
#         'Pays',
#         'Length\nSAP',
#         'Postal\nCode\nrequired',
#         'Nom\nPays',
#         'Nbr caract\nN° VAT\nOPALE',
#         'Length\nSAP P11',
#         'VAT nb mandatory if Europe ctry',
#         'NAF',
#         'Telephone'
#
#     ],
#     axis=1
# )
# namecols = ['name1', 'name2']
# newright['name'] = newright.apply(
#     lambda r: preprocessing.concatenate_names(r[namecols]),
#     axis=1
# )
# newright.drop(namecols, axis=1, inplace=True)
# del namecols
#
# streetcols = ['street1', 'street2']
# newright['street'] = newright.apply(
#     lambda r: preprocessing.concatenate_names(r[streetcols]),
#     axis=1
# )
# newright['siret'] = newright['siret'].apply(lambda r: preprocessing.idtostr(r))
# newright['siren'] = newright['siret'].apply(lambda r: None if pd.isnull(r) else r[:9])
# newright.drop(streetcols, axis=1, inplace=True)
# newright.set_index([ixname], inplace=True)
# newright.to_csv('P11.csv', index=True, sep='|', encoding='utf-8')
