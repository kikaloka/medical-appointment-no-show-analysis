"""Independent consistency/privacy checks; no row-level output is printed or saved."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analysis import load_and_prepare, EXPECTED_SHA, RAW

def main():
    checks=[]
    def check(condition,label):
        if not condition: raise AssertionError(label)
        checks.append(label)
    raw,df,audit=load_and_prepare()
    s=json.loads((ROOT/'results/summary.json').read_text())
    check(hashlib.sha256(RAW.read_bytes()).hexdigest()==EXPECTED_SHA,'original checksum preserved')
    check(raw.shape==(110527,14) and len(df)==110521,'snapshot schema and cohort counts')
    check(s['analysis_appointments']==len(df) and s['missed']==int(df.missed.sum()),'overview counts')
    check(np.isclose(s['missed_rate'],df.missed.mean()),'overview rate')
    check(pd.read_csv(ROOT/'results/cleaning-audit.csv')['count'].tolist()==audit['count'].tolist(),'cleaning audit reconciliation')
    for filename,column in [('lead-time','lead_group'),('age-groups','age_group'),('sms-overall','sms_group')]:
        table=pd.read_csv(ROOT/f'results/{filename}.csv')
        for _,row in table.iterrows():
            subset=df.loc[df[column].astype(str)==row[column]]
            check(len(subset)==row.appointments and int(subset.missed.sum())==row.missed,f'{filename}: {row[column]} counts')
            check(np.isclose(subset.missed.mean(),row.rate,atol=1e-8),f'{filename}: {row[column]} rate')
        check(int(table.appointments.sum())==len(df),f'{filename}: denominators reconcile')
        if 'low_95' in table:
            check(((0<=table.low_95)&(table.low_95<=table.rate)&(table.rate<=table.high_95)&(table.high_95<=1)).all(),f'{filename}: interval bounds')
    sms=pd.read_csv(ROOT/'results/sms-by-lead.csv')
    check(len(sms)==8 and sms.appointments.min()>=100,'SMS comparison common support and public minimum counts')
    for _,row in sms.iterrows():
        subset=df[(df.lead_group.astype(str)==row.lead_group)&(df.SMS_received==row.SMS_received)]
        check(len(subset)==row.appointments and int(subset.missed.sum())==row.missed,'SMS stratum counts')
        check(np.isclose(subset.missed.mean(),row.rate,atol=1e-8),'SMS stratum rate')
    positive=df[df.lead_days>0]
    weights=positive.groupby('lead_group',observed=True).size()/len(positive)
    rates=positive.groupby(['lead_group','SMS_received'],observed=True).missed.mean().unstack()
    direct=(rates.mul(weights,axis=0)).sum()
    check(np.isclose(direct[1],s['standardized_sms_rate']) and np.isclose(direct[0],s['standardized_no_sms_rate']),'standardization independently recomputed')
    check(np.isclose(100*(direct[1]-direct[0]),s['standardized_difference_pp']),'standardized percentage-point difference')
    first=df.sort_values(['scheduled','appointment','AppointmentID'],kind='stable').drop_duplicates('PatientId')
    sensitivity=pd.read_csv(ROOT/'results/sensitivity.csv')
    expected=[df,first,df[df.Age<=100]]
    for i,part in enumerate(expected):
        check(sensitivity.iloc[i].appointments==len(part) and np.isclose(sensitivity.iloc[i].rate,part.missed.mean(),atol=1e-8),'sensitivity cohort and rate')
    first_table=pd.read_csv(ROOT/'results/first-appointment-lead.csv')
    for _,row in first_table.iterrows():
        part=first[first.lead_group.astype(str)==row.lead_group]
        check(len(part)==row.appointments and np.isclose(part.missed.mean(),row.rate,atol=1e-8),'first-appointment lead comparison')
    for file in (ROOT/'results').glob('*.csv'):
        table=pd.read_csv(file)
        check(not {'PatientId','AppointmentID','ScheduledDay','AppointmentDay','Neighbourhood'} & set(table.columns),f'{file.name}: no identifier/record fields')
        check(len(table)<100,f'{file.name}: aggregate-only table size')
        if 'appointments' in table:check(table.appointments.min()>=100,f'{file.name}: minimum denominator')
    nb=nbformat.read(ROOT/'notebooks/appointment_attendance.ipynb',as_version=4)
    nbformat.validate(nb)
    texts=[]
    cells=[c for c in nb.cells if c.cell_type=='code']
    check(all(c.execution_count is not None for c in cells),'all notebook code cells freshly executed')
    for cell in cells:
        for output in cell.outputs:
            check(output.output_type!='error','notebook contains no errors')
            texts.append(output.get('text',''))
            texts.extend(v for k,v in output.get('data',{}).items() if k.startswith('text/'))
    tokens=set(re.findall(r'(?<!\d)\d{7,}(?!\d)',' '.join(texts)))
    ids=set(raw.PatientId.str.replace(r'\.0$','',regex=True)) | set(raw.AppointmentID)
    check(not tokens & ids,'no source identifier tokens in notebook text outputs')
    expected_files=['results/notebook-report.html']+[f'results/figures/{name}' for name in ['01-lead-time.png','02-age-groups.png','03-sms-within-lead.png','04-distributions.png']]
    for name in expected_files:check((ROOT/name).stat().st_size>1000,f'{name}: artifact present')
    # File links only; external URLs and in-document anchors are not fetched.
    for file in list(ROOT.glob('*.md'))+list((ROOT/'docs').glob('*.md')):
        text=file.read_text(encoding='utf-8-sig')
        links=re.findall(r'\]\(([^)]+)\)',text) if file.suffix=='.md' else re.findall(r'href="([^"]+)"',text)
        for link in links:
            if '://' in link or link.startswith('#'):continue
            target=(file.parent/link.split('#')[0]).resolve()
            if target==ROOT/'results/validation.json':continue
            check(target.exists(),f'{file.name}: local link {link}')
    readme=(ROOT/'README.md').read_text(encoding='utf-8-sig')
    for value in ['110,521','20.19','4.65','33.00','26.88','30.85']:
        check(value in readme,f'README key number {value}')
    if (ROOT/'.git').exists():
        for name in ['data/original/noshowappointments-kagglev2-may-2016.csv','private-originals/Investigate_a_Dataset.ipynb','.runtime/test.json']:
            r=subprocess.run(['git','check-ignore',name],cwd=ROOT,capture_output=True)
            check(r.returncode==0,f'Git excludes {name}')
    result={'status':'PASS','checks_passed':len(checks),'checks':checks,'source_sha256':EXPECTED_SHA,
            'notebook_code_cells_executed':len(cells),'scope':'Standalone analysis; aggregate outputs only'}
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(checks)} checks; fresh notebook cells: {len(cells)}; no raw records exported.')

if __name__=='__main__':main()
