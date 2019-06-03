Data Folders:

1. overnight-lf : Original dataset available at https://worksheets.codalab.org/bundles/0x7a8a1318d547444e9657341908fe24ea

2. normalized-data: Pre-processed data with entity anonymization and syntax truncation

3. scripts:
>  - vocabulay.py: Generates encoder and decoder vocabulary
>  - entity_dictionary: List of anonymized entities for each domain 
>  - entity_deanonymization.py: Uses entity_dictionary to replace the anonymized entities with corresponding values
