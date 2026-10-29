# Data requirements

Use the 2022 product-by-product release of EXIOBASE 3.9.6. The expected dimensions are 49 regions, 200 products per region and seven final-demand categories per region: 9,800 production nodes and 343 final-demand columns.

Keep the extracted source layout, including `metadata.json`, `file_parameters.json`, the MRIO matrices and these satellite accounts:

| Files | Content | Unit |
| --- | --- | --- |
| `A.txt`, `Z.txt`, `Y.txt`, `x.txt`, `unit.txt` | Technical coefficients, transactions, final demand and output | Coefficients or million EUR |
| `air_emissions/F.txt` and associated account files | Carbon dioxide emissions | kg CO2 |
| `employment/F.txt` and associated account files | Employment and labour hours by skill and sex | Thousand persons and million hours |
| `factor_inputs/F.txt` and associated account files | Compensation and other primary inputs | Million EUR |

The loader uses `pymrio.parse_exiobase3` and retains the source product and region ordering. Wage comparisons by skill use the three compensation rows labelled Low-, Medium- and High-skilled. Descriptive summaries also retain the broader wage total.

Carbon assigned to skills uses employment shares within each producing product cell. A cell with no positive skill-employment denominator retains its unassigned carbon in the aggregate total. Sensitivity analyses use labour-hour and wage shares as alternative allocation bases.

Set `EXIOBASE_PXP_PATH`, or pass `--data-path /path/to/IOT_2022_pxp` to the analysis script.
