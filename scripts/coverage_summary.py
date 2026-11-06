"""Quantify observed endpoint coverage of the constituent-model eligibility rule."""
from pathlib import Path
import sys
if __name__ == "__main__" and ("--help" in sys.argv or "-h" in sys.argv):
    print(__doc__)
    raise SystemExit(0)
import json
import pandas as pd

root=Path(__file__).resolve().parents[1]
concentration=pd.read_csv(root/'data_processed/conc_wide.csv').set_index('site_id')
status=pd.read_csv(root/'data_processed/api_status.csv').set_index('site_id')
benchmark=pd.read_csv(root/'data_processed/compounds.csv').set_index('compound').PNEC_lowest
detections=status.isin(['QUANT','EST']).sum()
modeled=detections[detections>=5].index
omitted=detections[detections<5].index
ratios=concentration.div(benchmark)
omitted_positive=ratios[omitted].gt(1).any(axis=1)
modeled_positive=ratios[modeled].gt(1).any(axis=1)
out={
    'sites':len(concentration),
    'modeled_apis':len(modeled),
    'modeled_with_benchmark':int(benchmark[modeled].notna().sum()),
    'omitted_apis':len(omitted),
    'omitted_with_benchmark':int(benchmark[omitted].notna().sum()),
    'omitted_survey_wide_concentration_share':float(concentration[omitted].sum().sum()/concentration.sum().sum()),
    'omitted_survey_wide_summed_ratio_share':float(ratios[omitted].sum().sum()/ratios.sum().sum()),
    'sites_with_omitted_api_exceedance':int(omitted_positive.sum()),
    'sites_positive_only_through_omitted_apis':int((omitted_positive&~modeled_positive).sum()),
    'per_omitted_api_exceedance_counts':{k:int(v) for k,v in ratios[omitted].gt(1).sum().items()},
    'convention':'Reported observed concentrations; nondetects and missing records contribute zero; this is not latent below-limit coverage.'
}
dest=root/'results/coverage_summary.json'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
